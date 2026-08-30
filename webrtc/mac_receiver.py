#!/usr/bin/env python3
"""mac 端 WebRTC 接收测试客户端（模拟 teleoprtc Client）"""
import asyncio, json, sys
from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription, RTCIceCandidate
import websockets

SERVER = sys.argv[1] if len(sys.argv) > 1 else "192.168.2.100"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 20

async def main():
    ws = await websockets.connect(f"ws://{SERVER}:8081")
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    received = 0
    bytes_total = 0

    audio_frames = [0]

    @pc.on("track")
    def on_track(track):
        print(f"[client] 收到远端 track: {track.kind}")
        if track.kind == "audio":
            async def consume_audio():
                while True:
                    try:
                        f = await track.recv()
                        audio_frames[0] += 1
                        if audio_frames[0] % 50 == 0:
                            print(f"[client] 音频帧: {audio_frames[0]}  {f.sample_rate}Hz {f.layout}", flush=True)
                    except Exception:
                        break
            asyncio.ensure_future(consume_audio())
            return
        async def consume():
            nonlocal received, bytes_total
            t0 = asyncio.get_event_loop().time()
            while True:
                try:
                    frame = await track.recv()
                    received += 1
                    bytes_total += (frame.width * frame.height * 3 // 2)
                    now = asyncio.get_event_loop().time()
                    if received in (1, 30, 60) or (now - t0) > 10:
                        fps = received / (now - t0)
                        print(f"[client] 已收 {received} 帧  {frame.width}x{frame.height} {frame.format}  "
                              f"FPS≈{fps:.1f}", flush=True)
                except Exception as e:
                    import traceback
                    print(f"[client] 接收异常: {type(e).__name__}: {repr(e)}")
                    traceback.print_exc()
                    break
        asyncio.ensure_future(consume())

    @pc.on("connectionstatechange")
    def on_state():
        print(f"[client] 连接状态: {pc.connectionState}")

    await ws.send(json.dumps({"type": "join"}))

    async def signaling():
        async for raw in ws:
            msg = json.loads(raw)
            t = msg["type"]
            if t == "offer":
                print(f"[client] 收到 offer ({len(msg['sdp'])} 字节)")
                await pc.setRemoteDescription(RTCSessionDescription(sdp=msg["sdp"], type="offer"))
                ans = await pc.createAnswer()
                await pc.setLocalDescription(ans)
                await ws.send(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))
                print("[client] 已发送 answer")
            elif t == "ice":
                cand = RTCIceCandidate(candidate=msg["candidate"], sdpMid="0", sdpMLineIndex=msg.get("mlineindex", 0))
                try:
                    await pc.addIceCandidate(cand)
                except Exception as e:
                    print(f"[client] ICE 添加失败: {e}")

    sig_task = asyncio.ensure_future(signaling())
    await asyncio.sleep(DURATION)
    print(f"[client] 测试结束: 共接收 {received} 视频帧, {audio_frames[0]} 音频帧")
    await pc.close()
    sig_task.cancel()

asyncio.run(main())
