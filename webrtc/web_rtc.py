#!/usr/bin/env python3
"""
web_rtc.py — RK3576 远端图像传输服务
参考 commaai/teleoprtc 架构：WebRTC 视频推流 + WebSocket 信令

链路：video11 (NV12) → videoscale(720p) → mpph264enc(硬编码) → webrtcbin → 浏览器
信令：WebSocket 服务器（端口 8080），JSON 消息 {type: offer/answer/ice/join}
浏览器端：/usr/local/bin/webrtc/index.html（http://<板子IP>:8080）

运行：sudo python3 web_rtc.py [端口=8080]
"""
import sys
import os
import json
import queue
import threading
import asyncio
import signal
import functools
import http.server
import websockets

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

SRC_W, SRC_H = 2688, 1520
OUT_W, OUT_H = 1280, 720

PIPELINE = f"""
v4l2src device=/dev/video11 !
video/x-raw,format=NV12,width={SRC_W},height={SRC_H},framerate=30/1 !
videoscale ! video/x-raw,width={OUT_W},height={OUT_H} !
mpph264enc rc-mode=cbr qp-init=26 !
h264parse !
rtph264pay pt=96 config-interval=1 !
queue name=q0 max-size-buffers=10
"""

class StaticHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

class RTCServer:
    def __init__(self, port=8080):
        self.port = port
        self.ws_port = port + 1      # WS 信令端口
        self.webroot = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.ws = None                  # 当前浏览器连接
        self.pipeline = Gst.parse_launch(PIPELINE)
        # 手动添加 webrtcbin 并链接 request sink pad（parse_launch 无法自动链 request pad）
        self.wbin = Gst.ElementFactory.make("webrtcbin", "wbin")
        self.pipeline.add(self.wbin)
        q0 = self.pipeline.get_by_name("q0")
        sinkpad = self.wbin.request_pad_simple("sink_%u")
        srcpad = q0.get_static_pad("src")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("链接 webrtcbin sink pad 失败")
        self.loop = None                # asyncio loop（WS 线程）
        self.gst_queue = queue.Queue()  # 主线程 -> gst 主循环
        self.running = True

        # webrtcbin 信号
        self.wbin.connect("on-negotiation-needed", self.on_neg_needed)
        self.wbin.connect("on-ice-candidate", self.on_ice_candidate)

        # bus 消息（answer / ice）
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)

    # ---------- GStreamer 回调（主线程 GLib loop） ----------
    def on_neg_needed(self, element):
        promise = Gst.Promise.new_with_change_func(self.on_offer_created)
        element.emit("create-offer", None, promise)

    def on_offer_created(self, promise):
        try:
            reply = promise.get_reply()
            offer = reply.get_value("offer")
            sdp = offer.sdp.as_text()
            print(f"[rtc] 生成 offer ({len(sdp)} 字节)")
            self.send_to_peer({"type": "offer", "sdp": sdp})
        except Exception as e:
            print(f"[rtc] offer 创建失败: {e}")

    def on_ice_candidate(self, element, mlineindex, candidate):
        print(f"[rtc] ICE 候选 mline={mlineindex}: {candidate}")
        self.send_to_peer({"type": "ice", "mlineindex": mlineindex, "candidate": candidate})

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ELEMENT:
            if message.has_name("GstWebRTCBin"):
                struct = message.get_structure()
                if struct.has_field("answer"):
                    answer = struct.get_value("answer")
                    sdp = answer.sdp.as_text()
                    print(f"[rtc] 收到 answer ({len(sdp)} 字节)")
                elif struct.has_field("candidate"):
                    mlineindex = struct.get_value("candidate").get_uint("mlineIndex")
                    candidate = struct.get_value("candidate").get_string("candidate")
                    print(f"[rtc] 远端 ICE mline={mlineindex}")
        elif message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[rtc] 管道错误: {err.message}\n  {dbg}")
            self.stop()

    def send_to_peer(self, msg):
        """主线程 -> WS 线程发送"""
        if self.ws and self.loop:
            def _send():
                try:
                    asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(msg)), self.loop)
                except Exception as e:
                    print(f"[rtc] 发送失败: {e}")
            self.loop.call_soon_threadsafe(_send)

    def gst_set_remote_answer(self, sdp):
        self.gst_queue.put(("answer", sdp))

    def gst_add_ice(self, candidate):
        self.gst_queue.put(("ice", candidate))

    def process_gst_queue(self):
        """在主线程 GLib loop 中处理来自 WS 的命令"""
        try:
            while True:
                kind, data = self.gst_queue.get_nowait()
                if kind == "answer":
                    self._apply_answer(data)
                elif kind == "ice":
                    self._apply_ice(data)
        except queue.Empty:
            pass
        return True

    def _apply_answer(self, sdp):
        try:
            answer = Gst.WebRTCSessionDescription.new(Gst.WebRTCSDPType.ANSWER, Gst.Sdp.SDP.new_from_text(sdp))
            self.wbin.emit("set-remote-description", answer)
            print("[rtc] 已应用远端 answer")
        except Exception as e:
            print(f"[rtc] 应用 answer 失败: {e}")

    def _apply_ice(self, candidate_str):
        try:
            self.wbin.emit("add-ice-candidate", 0, candidate_str)
            print(f"[rtc] 已添加远端 ICE: {candidate_str[:60]}")
        except Exception as e:
            print(f"[rtc] 添加 ICE 失败: {e}")

    # ---------- WebSocket 信令 ----------
    async def ws_handler(self, websocket):
        print(f"[ws] 浏览器连接: {websocket.remote_address}")
        self.ws = websocket
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                print(f"[ws] 收到消息: {t}")
                if t == "join":
                    # 触发协商，生成 offer
                    GLib.idle_add(self._trigger_negotiation)
                elif t == "answer":
                    self.gst_set_remote_answer(msg["sdp"])
                elif t == "ice":
                    self.gst_add_ice(msg["candidate"])
        except websockets.exceptions.ConnectionClosed:
            print("[ws] 浏览器断开")
        self.ws = None

    def _trigger_negotiation(self):
        self.wbin.emit("create-offer", None,
                       Gst.Promise.new_with_change_func(self.on_offer_created))
        return False

    async def ws_server(self):
        async with websockets.serve(self.ws_handler, "0.0.0.0", self.ws_port):
            print(f"[ws] 信令服务器监听 0.0.0.0:{self.ws_port}")
            await asyncio.Future()  # 永远运行

    # ---------- 主循环 ----------
    def run(self):
        # 启动 HTTP 静态服务器（index.html）
        def http_thread():
            handler = functools.partial(StaticHandler, directory=self.webroot)
            httpd = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), handler)
            print(f"[http] 页面服务器 http://0.0.0.0:{self.port}")
            httpd.serve_forever()
        threading.Thread(target=http_thread, daemon=True).start()

        # 启动 WS 信令线程
        def ws_thread():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.ws_server())

        threading.Thread(target=ws_thread, daemon=True).start()

        # gst 主循环
        GLib.timeout_add(100, self.process_gst_queue)
        print("[rtc] 启动管道...")
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f"[rtc] 运行中。浏览器访问 http://<本机IP>:{self.port}")
        GLib.MainLoop().run()

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    srv = RTCServer(port)

    def on_sig(signum, frame):
        print("\n[rtc] 退出...")
        srv.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    srv.run()
