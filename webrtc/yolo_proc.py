#!/usr/bin/env python3
"""
yolo_proc.py — 独立 YOLO 检测进程（进程分离，消除 GIL 竞争）
主进程把视频帧写入 /dev/shm/camframe.yuv（I420 720p），本进程检测后写结果 JSON。

帧传递：/dev/shm/camframe.yuv（原子写：tmp+rename）
结果传递：/dev/shm/detect.json
"""
import os
import sys
import time
import json
import numpy as np
import cv2

SHM = "/dev/shm"
FRAME_PATH = os.path.join(SHM, "camframe.yuv")
RESULT_PATH = os.path.join(SHM, "detect.json")

sys.path.insert(0, "/usr/local/bin/webrtc")
from yolo_detect import YoloDetector

MODEL = "/usr/local/bin/webrtc/yolo11n_coco.rknn"
OUT_W, OUT_H = 1280, 720
FRAME_SIZE = OUT_W * OUT_H * 3 // 2


def main():
    print("[yolo-proc] 启动", flush=True)
    det = YoloDetector(MODEL)
    print("[yolo-proc] 模型就绪", flush=True)

    last_mtime = 0
    while True:
        try:
            st = os.stat(FRAME_PATH)
        except FileNotFoundError:
            time.sleep(0.1)
            continue
        if st.st_mtime == last_mtime or st.st_size != FRAME_SIZE:
            time.sleep(0.02)
            continue
        last_mtime = st.st_mtime

        # 读帧（I420 720p）
        with open(FRAME_PATH, "rb") as f:
            i420 = f.read(FRAME_SIZE)
        if len(i420) != FRAME_SIZE:
            continue

        t0 = time.time()
        img = cv2.cvtColor(
            np.frombuffer(i420, np.uint8).reshape(OUT_H * 3 // 2, OUT_W),
            cv2.COLOR_YUV2BGR_I420)
        objs = det.detect(img)
        det_ms = int((time.time() - t0) * 1000)

        result = {"objects": objs, "det_ms": det_ms, "ts": time.time()}
        # 原子写结果
        tmp = RESULT_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.rename(tmp, RESULT_PATH)

        if objs:
            names = ", ".join(f"{o['label']}:{o['conf']}" for o in objs[:4])
            print(f"[yolo-proc] {len(objs)} 目标 ({det_ms}ms): {names}", flush=True)
        elif int(time.time()) % 5 == 0:
            print(f"[yolo-proc] 检测中… 0 目标 ({det_ms}ms)", flush=True)


if __name__ == "__main__":
    main()
