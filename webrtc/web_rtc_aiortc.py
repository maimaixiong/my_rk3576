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
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)
import av
import numpy as np
import cv2
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription, VideoStreamTrack, AudioStreamTrack
from aiortc.mediastreams import VideoFrame, AudioFrame
import websockets
from v4l2_cap import V4L2Capture

SRC_W, SRC_H = 2688, 1520
OUT_W, OUT_H = 1280, 720   # 输出分辨率（720p）

PIPE = f"""
v4l2src device=/dev/video11 !
video/x-raw,format=NV12,width={SRC_W},height={SRC_H},framerate=30/1 !
videoscale ! video/x-raw,width={OUT_W},height={OUT_H} !
appsink name=sink sync=false drop=true max-buffers=2
"""

class StaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        # 禁止缓存（确保浏览器总是加载最新页面）
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

class CameraTrack(VideoStreamTrack):
    """V4L2 直读 → 软件缩放/AWB/AE → MPP H264 硬编（后台线程）→ aiortc 发送"""
    kind = "video"

    def __init__(self, cap: V4L2Capture):
        super().__init__()
        self.cap = cap
        self._fc = 0
        self._u_off = None
        self._v_off = None
        self._exp = 800
        self._gain = 128
        self._bright = 0
        self.q = queue.Queue(maxsize=2)

        self._sy = (np.arange(OUT_H) * SRC_H // OUT_H)
        self._sx = (np.arange(OUT_W) * SRC_W // OUT_W)
        self._usy = (np.arange(OUT_H // 2) * (SRC_H // 2) // (OUT_H // 2))
        self._usx = (np.arange(OUT_W // 2) * (SRC_W // 2) // (OUT_W // 2))

        # gst MPP 硬编码管道
        pipe_str = f"""
appsrc name=src format=time is-live=true max-buffers=4 !
video/x-raw,format=I420,width={OUT_W},height={OUT_H},framerate=30/1 !
mpph264enc rc-mode=cbr qp-init=24 profile=baseline level=31 gop=30 header-mode=each-idr !
h264parse !
appsink name=sink sync=false drop=true max-buffers=3
"""
        self.pipe = Gst.parse_launch(pipe_str)
        self.gsrc = self.pipe.get_by_name("src")
        self.gsink = self.pipe.get_by_name("sink")
        self.pipe.set_state(Gst.State.PLAYING)
        print(f"[cam] MPP H264 硬编就绪 {OUT_W}x{OUT_H}", flush=True)

        self._fail_count = 0
        self._stop = False
        threading.Thread(target=self._enc_loop, daemon=True).start()

    def close(self):
        """停止编码线程并释放资源（连接断开/重建时）"""
        self._stop = True
        try:
            self.pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            self.cap.close()
        except Exception:
            pass

    def _frame_i420(self):
        """V4L2 抓帧 + cv2 缩放 + AWB → I420 bytes（cv2 释放 GIL）"""
        data = self.cap.grab()
        y = np.frombuffer(data[:SRC_W * SRC_H], np.uint8).reshape(SRC_H, SRC_W)
        uv = np.frombuffer(data[SRC_W * SRC_H:], np.uint8).reshape(SRC_H // 2, SRC_W)

        y_s = cv2.resize(y, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        u_s = cv2.resize(uv[:, 0::2], (OUT_W // 2, OUT_H // 2),
                         interpolation=cv2.INTER_LINEAR)
        v_s = cv2.resize(uv[:, 1::2], (OUT_W // 2, OUT_H // 2),
                         interpolation=cv2.INTER_LINEAR)

        # AWB（灰度世界，降采样均值；cv2 加减补偿释放 GIL）
        if self._fc % 5 == 0:
            um = float(u_s[::4, ::4].mean()) - 128.0
            vm = float(v_s[::4, ::4].mean()) - 128.0
            if self._u_off is None:
                self._u_off, self._v_off = um, vm
            else:
                self._u_off = 0.85 * self._u_off + 0.15 * um
                self._v_off = 0.85 * self._v_off + 0.15 * vm

        if abs(self._u_off) > 1:
            off = int(round(min(abs(self._u_off), 255)))
            if self._u_off > 0:
                u_s = cv2.subtract(u_s, off)
            else:
                u_s = cv2.add(u_s, off)
        if abs(self._v_off) > 1:
            off = int(round(min(abs(self._v_off), 255)))
            if self._v_off > 0:
                v_s = cv2.subtract(v_s, off)
            else:
                v_s = cv2.add(v_s, off)
        return y_s.tobytes() + u_s.tobytes() + v_s.tobytes()

    def _ae_adjust(self):
        import subprocess as _sp
        TARGET, DEAD = 60.0, 5.0
        EXP_MIN, EXP_MAX50 = 200, 972
        GAIN_MIN, GAIN_MAX = 128, 1984
        err = TARGET - self._bright
        if err > DEAD:
            if self._exp < EXP_MAX50:
                self._exp = min(self._exp + 80, EXP_MAX50)
            elif self._gain < GAIN_MAX:
                self._gain = min(int(self._gain * 1.2), GAIN_MAX)
        elif err < -DEAD:
            if self._gain > GAIN_MIN:
                self._gain = max(int(self._gain / 1.2), GAIN_MIN)
            elif self._exp > EXP_MIN:
                self._exp = max(self._exp - 80, EXP_MIN)
        try:
            _sp.run(["v4l2-ctl", "-d", "/dev/v4l-subdev3",
                     "--set-ctrl", f"exposure={self._exp},analogue_gain={self._gain}"],
                    capture_output=True, timeout=2)
        except Exception:
            pass

    def _reinit(self):
        """重建 V4L2 采集 + gst 硬编管道（流中断恢复）"""
        import time as _t
        try:
            self.pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            self.cap.close()
        except Exception:
            pass
        _t.sleep(0.5)
        self.cap = V4L2Capture()
        self.cap.open()
        self.pipe.set_state(Gst.State.PLAYING)
        _t.sleep(0.3)

    def _enc_loop(self):
        """后台编码线程：抓帧→缩放→硬编→入队 H264 包"""
        import time as _t
        while not self._stop:
            try:
                i420 = self._frame_i420()
                if self._fc % 15 == 0:
                    self._bright = 0.7 * self._bright + 0.3 * float(
                        np.frombuffer(i420[:OUT_W * OUT_H], np.uint8).mean())
                    self._ae_adjust()

                pts = self._fc * (Gst.SECOND // 30)
                buf = Gst.Buffer.new_allocate(None, len(i420), None)
                buf.fill(0, i420)
                buf.pts = buf.dts = pts
                buf.duration = Gst.SECOND // 30
                self.gsrc.emit("push-buffer", buf)

                h264 = None
                for _ in range(50):
                    sample = self.gsink.emit("try-pull-sample", 0)
                    if sample:
                        b = sample.get_buffer()
                        ok, mi = b.map(Gst.MapFlags.READ)
                        h264 = bytes(mi.data)
                        b.unmap(mi)
                        break
                    _t.sleep(0.001)

                if h264:
                    if self.q.full():
                        try: self.q.get_nowait()
                        except queue.Empty: pass
                    self.q.put(h264)
                self._fc += 1
                self._fail_count = 0
            except Exception as e:
                if self._stop:
                    return                      # 已关闭，退出线程
                self._fail_count += 1
                if self._fail_count % 5 == 0:
                    print(f"[cam] 编码线程错误: {type(e).__name__}: {e} (连续 {self._fail_count})", flush=True)
                if self._fail_count >= 10:      # 连续失败 ~200ms → 重建视频链路
                    print("[cam] ⚠️ 视频链路重建...", flush=True)
                    try:
                        self._reinit()
                        print("[cam] ✅ 视频链路已恢复", flush=True)
                    except Exception as e2:
                        print(f"[cam] 重建失败: {e2}", flush=True)
                    self._fail_count = 0
                _t.sleep(0.02)

    async def recv(self):
        from aiortc.mediastreams import MediaStreamError
        if not hasattr(self, "_sent"):
            self._sent = 0
        self._sent += 1
        if self._sent % 120 == 0:
            print(f"[cam] 已发送 {self._sent} 帧 (RTP)", flush=True)
        while True:
            try:
                h264 = self.q.get_nowait()
                break
            except queue.Empty:
                await asyncio.sleep(0.002)
        packet = av.Packet(h264)
        packet.pts = self._fc
        packet.time_base = fractions.Fraction(1, 30)
        self._fc += 1
        return packet


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

        # 音频电平检测（每 5 帧 ~100ms，实时声压）
        if self._pts % (self.samples * 5) == 0:
            import numpy as _np
            rms = float(_np.sqrt(_np.mean(
                _np.frombuffer(data, _np.int16).astype(_np.float64) ** 2)))
            if self.on_level:
                self.on_level(rms)

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
            if hasattr(self, "track") and self.track:
                try:
                    self.track.close()
                except Exception:
                    pass
            if hasattr(self, "mic") and self.mic:
                try:
                    self.mic.close()
                except Exception:
                    pass

    async def handle_join(self):
        """新连接：建 PC + 视频轨 + 发起 offer"""
        if self.pc:
            await self.pc.close()
        if hasattr(self, "track") and self.track:
            try:
                self.track.close()    # 停旧编码线程 + 释放 V4L2
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
        # SDP 编辑：只留 H264（删 VP8/97/98），并加码率限制
        lines = offer.sdp.split("\r\n")
        out = []
        for l in lines:
            # 删 VP8 相关行
            if ("rtpmap:97 " in l or "rtpmap:98 " in l or "apt=97" in l
                    or "apt=98" in l or "fmtp:97" in l or "fmtp:98" in l):
                continue
            out.append(l)
            if l.startswith("m=video"):
                out.append("b=AS:3000")
                # m 行只保留 H264 payload（99/100/101/102）
        # 修正 m=video 行：删 97 98
        for i, l in enumerate(out):
            if l.startswith("m=video"):
                parts = l.split()
                parts = [p for p in parts if p not in ("97", "98")]
                out[i] = " ".join(parts)
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
        """声压回调：RMS → dBFS，推给浏览器显示"""
        import math
        db = 20 * math.log10(max(rms, 0.5) / 32768.0)
        if rms < 5:
            print(f"[mic] ⚠️ 麦克风无信号 (RMS={rms:.1f}, {db:.0f}dB) — 请确认外接麦克风", flush=True)
        elif self.ws:
            asyncio.run_coroutine_threadsafe(
                self._send({"type": "audio_meter", "db": round(db, 1), "rms": round(rms, 1)}),
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
