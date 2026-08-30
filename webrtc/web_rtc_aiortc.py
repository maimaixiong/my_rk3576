#!/usr/bin/env python3
"""
web_rtc_aiortc.py — RK3576 远端图像传输服务（aiortc 版）
参考 commaai/teleoprtc 架构：VideoStreamTrack + RTCPeerConnection + WebSocket 信令

链路：video11 (NV12 2688×1520) → gst appsink (720p) → aiortc 编码 → WebRTC → 浏览器
信令：WebSocket 端口 8081，JSON {type: join/offer/answer/ice}
页面：HTTP 端口 8080 → index.html

运行：sudo python3 web_rtc_aiortc.py [http端口=8080]
"""
import sys
import os
import json
import queue
import threading
import asyncio
import fractions
import signal
import functools
import http.server

import ssl
import subprocess
import numpy as np
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from aiortc.mediastreams import VideoFrame, AudioFrame
import websockets
from v4l2_cap import V4L2Capture

SRC_W, SRC_H = 2688, 1520
OUT_W, OUT_H = 640, 360   # 输出分辨率（可调；1280x720 时 VP8 软编仅 ~10fps）

PIPE = f"""
v4l2src device=/dev/video11 !
video/x-raw,format=NV12,width={SRC_W},height={SRC_H},framerate=30/1 !
videoscale ! video/x-raw,width={OUT_W},height={OUT_H} !
appsink name=sink sync=false drop=true max-buffers=2
"""

class StaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

class CameraTrack(VideoStreamTrack):
    """V4L2 直读 + numpy 缩放的 WebRTC 视频轨（参考 teleoprtc VideoTrack）"""
    kind = "video"

    def __init__(self, cap: V4L2Capture):
        super().__init__()
        self.cap = cap
        self._fc = 0
        self._u_off = None      # AWB 色偏补偿（IIR 平滑）
        self._v_off = None
        self._exp = 800         # 软件 AE 当前曝光
        self._gain = 128
        self._bright = 0
        # 预计算缩放索引（最近邻）
        self._sy = (np.arange(OUT_H) * SRC_H // OUT_H)
        self._sx = (np.arange(OUT_W) * SRC_W // OUT_W)
        # UV 分离后平面 (H/2, W/2)：行数 OUT_H//2，列数 OUT_W//2（各取一半）
        self._usy = (np.arange(OUT_H // 2) * (SRC_H // 2) // (OUT_H // 2))
        self._usx = (np.arange(OUT_W // 2) * (SRC_W // 2) // (OUT_W // 2))

    def _frame_data(self):
        """取帧 + NV12→yuv420p 缩放（在 executor 线程执行，释放 GIL）"""
        data = self.cap.grab()
        y = np.frombuffer(data[:SRC_W * SRC_H], np.uint8).reshape(SRC_H, SRC_W)
        uv = np.frombuffer(data[SRC_W * SRC_H:], np.uint8).reshape(SRC_H // 2, SRC_W)

        y_s_arr = y[self._sy][:, self._sx]
        y_s = y_s_arr.tobytes()
        # 先分离 U/V 平面（NV12 交错：偶列 U，奇列 V），再分别缩放（修复红绿条纹）
        u_c = uv[self._usy][:, 0::2][:, self._usx].astype(np.int16) - 128
        v_c = uv[self._usy][:, 1::2][:, self._usx].astype(np.int16) - 128

        # 软件 AE（anti-flicker 50Hz：曝光上限 972 行≈20ms=光源整数周期，之后用增益）
        if self._fc % 5 == 0:
            self._bright = 0.7 * self._bright + 0.3 * float(y_s_arr.mean())
            self._ae_adjust()

        # 软件 AWB：灰度世界（每 5 帧更新一次平滑色偏）
        if self._fc % 5 == 0:
            u_mean, v_mean = float(u_c.mean()), float(v_c.mean())
            if self._u_off is None:
                self._u_off, self._v_off = u_mean, v_mean
            else:
                self._u_off = 0.85 * self._u_off + 0.15 * u_mean
                self._v_off = 0.85 * self._v_off + 0.15 * v_mean

        u_s = np.ascontiguousarray(np.clip(u_c - self._u_off + 128, 0, 255).astype(np.uint8)).tobytes()
        v_s = np.ascontiguousarray(np.clip(v_c - self._v_off + 128, 0, 255).astype(np.uint8)).tobytes()
        return y_s, u_s, v_s

    def _ae_adjust(self):
        """曝光/增益闭环，曝光对齐 50Hz 防条纹"""
        import subprocess
        TARGET, DEAD = 60.0, 5.0
        EXP_MIN, EXP_MAX50 = 200, 972    # 上限 972≈20ms（50Hz 整数周期）
        GAIN_MIN, GAIN_MAX = 128, 1984
        err = TARGET - self._bright
        if err > DEAD:                     # 暗
            if self._exp < EXP_MAX50:
                self._exp = min(self._exp + 80, EXP_MAX50)
            elif self._gain < GAIN_MAX:
                self._gain = min(int(self._gain * 1.2), GAIN_MAX)
        elif err < -DEAD:                  # 亮
            if self._gain > GAIN_MIN:
                self._gain = max(int(self._gain / 1.2), GAIN_MIN)
            elif self._exp > EXP_MIN:
                self._exp = max(self._exp - 80, EXP_MIN)
        try:
            subprocess.run(["v4l2-ctl", "-d", "/dev/v4l-subdev3",
                            "--set-ctrl", f"exposure={self._exp},analogue_gain={self._gain}"],
                           capture_output=True, timeout=2)
        except Exception:
            pass

    async def recv(self):
        try:
            y, u, v = await asyncio.get_event_loop().run_in_executor(None, self._frame_data)
        except Exception as e:
            from aiortc.mediastreams import MediaStreamError
            print(f"[cam] 取帧失败: {e}", flush=True)
            raise MediaStreamError("取帧失败")

        frame = VideoFrame(width=OUT_W, height=OUT_H, format="yuv420p")
        frame.planes[0].update(y)
        frame.planes[1].update(u)
        frame.planes[2].update(v)
        frame.pts = self._fc
        frame.time_base = fractions.Fraction(1, 30)
        self._fc += 1
        return frame


class MicTrack(AudioStreamTrack):
    """ES8388 麦克风 → Opus（arecord 子进程管道，20ms/帧）"""
    kind = "audio"

    def __init__(self, on_level=None):
        super().__init__()
        self.sample_rate = 48000
        self.channels = 2
        self.samples = 960                      # 20ms @ 48kHz
        self.frame_bytes = self.samples * self.channels * 2
        self._pts = 0
        self.on_level = on_level
        self.proc = subprocess.Popen(
            ["arecord", "-D", "hw:0,0", "-f", "S16_LE", "-r", "48000",
             "-c", "2", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def close(self):
        """释放音频设备（杀 arecord 进程）"""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()

    async def recv(self):
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.proc.stdout.read, self.frame_bytes)
        except Exception as e:
            print(f"[mic] 读取失败: {e}", flush=True)
            from aiortc.mediastreams import MediaStreamError
            raise MediaStreamError("麦克风读取失败")
        if len(data) < self.frame_bytes:
            from aiortc.mediastreams import MediaStreamError
            raise MediaStreamError("音频流结束")

        # 音频电平检测（每 25 帧 ~500ms 一次，无信号时提示）
        if self._pts % (self.samples * 25) == 0:
            import numpy as _np
            rms = _np.sqrt(_np.mean(
                _np.frombuffer(data, _np.int16).astype(_np.float64) ** 2))
            if self.on_level:
                self.on_level(float(rms))

        frame = AudioFrame(format="s16", layout="stereo", samples=self.samples)
        frame.planes[0].update(data)
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        frame.pts = self._pts
        self._pts += self.samples
        return frame


class RTCServer:
    def __init__(self, http_port=8080):
        self.http_port = http_port
        self.ws_port = http_port + 1
        self.webroot = os.path.dirname(os.path.abspath(__file__))
        self.pc = None
        self.ws = None
        self.loop = None

    # ---------- aiortc 信令 ----------
    async def ws_handler(self, websocket):
        print(f"[ws] 浏览器连接: {websocket.remote_address}")
        self.ws = websocket
        try:
            async for raw in websocket:
                msg = json.loads(raw)
                t = msg.get("type")
                print(f"[ws] 消息: {t}")
                if t == "join":
                    await self.handle_join()
                elif t == "answer":
                    await self.handle_answer(msg["sdp"])
                elif t == "ice":
                    await self.handle_ice(msg.get("candidate"), msg.get("mlineindex", 0))
        except websockets.exceptions.ConnectionClosed:
            print("[ws] 浏览器断开")
        finally:
            self.ws = None
            # 断开时释放设备（供下次连接）
            if hasattr(self, "mic") and self.mic:
                try:
                    self.mic.close()
                except Exception:
                    pass

    async def handle_join(self):
        """新连接：建 PC + 视频轨 + 发起 offer"""
        if self.pc:
            await self.pc.close()
        if hasattr(self, "cap") and self.cap:
            try:
                self.cap.close()      # 释放 V4L2 设备（否则下次 join EBUSY）
            except Exception:
                pass
        if hasattr(self, "mic") and self.mic:
            try:
                self.mic.close()      # 释放音频设备（否则下次 join 无声音）
            except Exception:
                pass

        # 局域网直连：禁用 STUN（板子 UDP 出网不通，避免 gather 卡 Google STUN 超时）
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))

        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))

        @self.pc.on("icegatheringstatechange")
        async def on_gather():
            print(f"[rtc] ICE gather 状态: {self.pc.iceGatheringState}")

        @self.pc.on("icecandidate")
        async def on_ice(candidate):
            if candidate is not None:
                print(f"[rtc] ICE 候选: {candidate.candidate[:60]}...")
            else:
                print("[rtc] ICE 候选收集完成")

        @self.pc.on("connectionstatechange")
        async def on_state():
            print(f"[rtc] 连接状态: {self.pc.connectionState}")

        @self.pc.on("track")
        async def on_track(track):
            print(f"[rtc] 远端 track: {track.kind}")
            if track.kind == "audio":
                # 浏览器 → 板子：播放到 ES8388 喇叭（双向对讲）
                import subprocess as _sp
                self._aplay = _sp.Popen(
                    ["aplay", "-D", "hw:0,0", "-f", "S16_LE", "-r", "48000",
                     "-c", "2", "-t", "raw", "-q"],
                    stdin=_sp.PIPE, stderr=_sp.DEVNULL)
                while True:
                    try:
                        f = await track.recv()
                        arr = f.to_ndarray()          # (channels, samples) int16
                        if f.layout == "stereo" and arr.shape[0] == 2:
                            data = arr.T.tobytes()    # 交错
                        else:
                            mono = arr[0]
                            import numpy as _np
                            data = _np.repeat(mono, 2).tobytes()
                        self._aplay.stdin.write(data)
                    except Exception as e:
                        print(f"[rtc] 远端音频结束: {e}", flush=True)
                        try:
                            self._aplay.stdin.close()
                        except Exception:
                            pass
                        break

        # 视频轨：V4L2 直读（30fps，比 gst v4l2src 快 3 倍）
        self.cap = V4L2Capture()
        self.cap.open()
        self.track = CameraTrack(self.cap)
        self.pc.addTrack(self.track)
        # 音频轨（ES8388 麦克风），带电平检测回调
        self.mic = MicTrack(on_level=self._audio_level)
        self.pc.addTrack(self.mic)
        print(f"[rtc] 摄像头+麦克风就绪，创建 offer...")

        offer = await self.pc.createOffer()
        # 限制码率：m=video 行加 b=AS（aiortc 解析为 target_bitrate）
        lines = offer.sdp.split("\r\n")
        out = []
        for l in lines:
            out.append(l)
            if l.startswith("m=video"):
                out.append("b=AS:2500")
        offer.sdp = "\r\n".join(out)
        await self.pc.setLocalDescription(offer)
        sdp_send = self.pc.localDescription.sdp   # 候选在 setLocalDescription 后写入这里
        n_cands = len([l for l in sdp_send.splitlines() if "candidate" in l])
        print(f"[rtc] offer 生成 ({len(sdp_send)} 字节, 候选 {n_cands} 条)")
        await self._send({"type": "offer", "sdp": sdp_send})

    async def handle_answer(self, sdp):
        if self.pc:
            await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))
            print("[rtc] 已设置远端 answer")

    async def handle_ice(self, candidate, mlineindex):
        if self.pc and candidate:
            from aiortc import RTCIceCandidate
            cand = RTCIceCandidate(candidate=candidate, sdpMid="0", sdpMLineIndex=mlineindex)
            await self.pc.addIceCandidate(cand)
            print("[rtc] 已添加远端 ICE")

    def _audio_level(self, rms):
        """音频电平回调：推给浏览器显示"""
        level = min(int(rms / 100), 10)
        if rms < 5:
            print(f"[mic] ⚠️ 麦克风无信号 (RMS={rms:.1f}) — 请确认外接麦克风", flush=True)
        elif self.ws:
            asyncio.run_coroutine_threadsafe(
                self._send({"type": "audio_level", "level": level}),
                asyncio.get_event_loop())

    async def _send(self, msg):
        if self.ws:
            try:
                await self.ws.send(json.dumps(msg))
            except Exception as e:
                print(f"[ws] 发送失败: {e}")

    async def ws_server(self, ssl_ctx=None):
        async with websockets.serve(self.ws_handler, "0.0.0.0", self.ws_port, ssl=ssl_ctx):
            print(f"[wss] 信令服务器 0.0.0.0:{self.ws_port}")
            await asyncio.Future()

    # ---------- 主入口 ----------
    def _ssl_ctx(self):
        """自签证书（浏览器首次访问需点'继续访问'）"""
        cert = os.path.join(self.webroot, "cert.pem")
        key = os.path.join(self.webroot, "key.pem")
        if not (os.path.exists(cert) and os.path.exists(key)):
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048",
                            "-keyout", key, "-out", cert, "-days", "365", "-nodes",
                            "-subj", "/CN=192.168.2.100"], capture_output=True)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        return ctx

    async def main(self, use_https=False):
        handler = functools.partial(StaticHandler, directory=self.webroot)
        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.http_port), handler)
        if use_https:
            ctx = self._ssl_ctx()
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[{'https' if use_https else 'http'}] 页面服务器 {'https' if use_https else 'http'}://0.0.0.0:{self.http_port}")

        await self.ws_server(self._ssl_ctx() if use_https else None)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = RTCServer(port)

    def on_sig(signum, frame):
        print("\n[rtc] 退出...")
        if srv.pc:
            asyncio.run(srv.pc.close())
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    use_https = "--https" in sys.argv
    asyncio.run(srv.main(use_https))
