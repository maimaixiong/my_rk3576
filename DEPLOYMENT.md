# RK3576 边缘 AI 设备部署文档

> 目标：在 RK3576（或同类 Rockchip 4GB 设备）上部署"摄像头+鱼眼+WebRTC+YOLO检测+AI对话"一体化系统
> 参考架构：commaai/teleoprtc（WebRTC）、ThinkOffApp/CarWatch（设备 Agent 分层）
> 适用：本仓库 `github.com/maimaixiong/my_rk3576`

## 1. 系统架构

```
┌── 主进程 web_rtc_aiortc.py ──────────────────────────────┐
│  V4L2(2688×1520) → 鱼眼去畸变 → MPP H264 → WebRTC → 浏览器 │
│  音频(ES8388) + 系统监控(温度/CPU/NPU/fps)                │
│  HTTP:8080 + HTTPS:8443 + WS:8081 + WSS:8084             │
│  每15帧 → /dev/shm/camframe.yuv（帧共享）                 │
├── 检测进程 yolo_proc.py（独立）───────────────────────────┤
│  读 shm 帧 → YOLO11n RKNN INT8 → detect.json             │
├── LLM 服务 llama-server（独立, llama.cpp）────────────────┤
│  Qwen2.5-1.5B GGUF（CPU）— 感知注入对话                   │
├── STT（惰性加载, 语音时）─────────────────────────────────┤
│  faster-whisper（webm → ffmpeg → 转录）                  │
└──────────────────────────────────────────────────────────┘
```

## 2. 硬件要求

| 项 | 要求 |
|---|---|
| SoC | RK3576（6 TOPS NPU，4×A72+4×A53） |
| 内存 | 4GB（LLM 1.5B + whisper 共存临界，3B 模型超限） |
| 存储 | eMMC ≥ 8GB 空闲（模型 + 4GB swap） |
| 摄像头 | MIPI CSI（本文档以 OS04C10 2688×1520 为例） |
| 音频 | ES8388 codec（采集需外接 mic；播放板载喇叭/耳机） |
| 网络 | USB 网卡 100M 或板载 GMAC |
| 主机 | Mac/Linux（部署用，需能访问板子） |

## 3. 前置准备（一次性）

### 3.1 板子系统
```bash
# Ubuntu 22.04 aarch64，内核含 rkcif/rkisp/mpp/rknpu 驱动
sudo apt update
sudo apt install -y ffmpeg gcc g++ cmake git v4l-utils python3-pip \
  python3-opencv gstreamer1.0-tools
```

### 3.2 免密登录（主机侧）
```bash
ssh-keygen -t ed25519   # 若无
ssh-copy-id myir@<板IP>
# ~/.ssh/config 添加：
# Host rk3576
#     HostName <板IP>
#     User myir
#     IdentityFile ~/.ssh/id_ed25519
```

### 3.3 网络（若板子需出网：apt/pip/git）
板子默认路由 + 代理（详见"故障排查-网络"）。本文示例：主机 192.168.2.1 跑 HTTP CONNECT 代理 :1081，板子代理走它。

## 4. Python 依赖（板子）

```bash
# 系统级安装（sudo -H，代理示例 http://192.168.2.1:1081）
sudo -H pip3 install aiortc websockets numpy opencv-python-headless

# RKNN 运行时（板载 /usr/lib/librknnrt.so 已有，用 ctypes 直调——无需 rknnlite）
# 语音 STT
sudo -H pip3 install faster-whisper
```

**关键**：本系统**不依赖 rknn-toolkit2/rknnlite**（推理用 ctypes 调 librknnrt）。

## 5. 文件部署

```bash
# 从仓库拷贝到板子（部署脚本见 deploy.sh 模式）
scp webrtc/*.py webrtc/index.html rk3576:/tmp/
ssh rk3576 "sudo mkdir -p /usr/local/bin/webrtc && sudo cp /tmp/*.py /tmp/index.html /usr/local/bin/webrtc/"
```

### 板子 `/usr/local/bin/webrtc/` 目录应含：
| 文件 | 作用 |
|---|---|
| `web_rtc_aiortc.py` | 主服务（WebRTC+视频+监控+对话+语音） |
| `yolo_proc.py` | 独立 YOLO 检测进程 |
| `yolo_detect.py` | YOLO 检测/后处理库 |
| `rknn_api_ctypes.py` | ctypes RKNN 推理封装 |
| `v4l2_cap.py` | V4L2 MPLANE 取帧（ctypes） |
| `index.html` | Web 页面（视频/监控/聊天/语音） |
| `whisper-base/` | faster-whisper base 模型（138MB，中文） |

### 模型文件（各自下载，>100MB 不入 git）
| 模型 | 位置 | 用途 |
|---|---|---|
| `yolo11n_coco.rknn` | `/usr/local/bin/webrtc/` | YOLO 检测（7.3MB，RK3576 INT8） |
| `qwen2.5-1.5b-q4.gguf` | 同上 | LLM 对话（1.07GB，CPU） |
| `whisper-base/*` | `whisper-base/` | 语音识别（138MB） |

下载源：
- YOLO rknn：转换自 yolov8n.onnx（需 rknn-toolkit2 + torch1.13 aarch64，详见"模型转换"）
- GGUF：`hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF`（Q4_K_M）
- whisper：`hf-mirror.com/Systran/faster-whisper-base`（model.bin+config.json+tokenizer.json+vocabulary.txt）

## 6. 系统服务（systemd）

### 6.1 web-rtc.service
```ini
[Unit]
Description=RK3576 WebRTC video streaming
After=multi-user.target

[Service]
KillMode=mixed
Type=simple
WorkingDirectory=/usr/local/bin/webrtc
ExecStart=/usr/bin/python3 -u /usr/local/bin/webrtc/web_rtc_aiortc.py 8080 --https
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### 6.2 llama-server.service（LLM）
```ini
[Unit]
Description=llama.cpp LLM server
After=multi-user.target

[Service]
Type=simple
WorkingDirectory=/usr/local/bin/webrtc
ExecStart=/tmp/llama.cpp/build/bin/llama-server -m qwen2.5-1.5b-q4.gguf --port 8082 --host 127.0.0.1 -c 1024 -t 8
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 6.3 llama.cpp 编译（板子）
```bash
cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8 --target llama-server
```

### 6.4 传感器参数固化（isp-fix）
```bash
# vertical_blanking=54（30fps；AE 曝光>486 行会拖低帧率）
cat > /usr/local/bin/isp-fix.sh << 'EOF'
#!/bin/bash
sleep 3
v4l2-ctl -d /dev/v4l-subdev3 --set-ctrl vertical_blanking=54 2>/dev/null
EOF
chmod +x /usr/local/bin/isp-fix.sh
# + systemd oneshot 开机执行
```

### 6.5 冲突服务处理
```bash
# camera-npu.service（若存在，抢 video11）必须禁用
sudo systemctl stop camera-npu && sudo systemctl disable camera-npu
# rkaiq_3A（os04c10 无 iqfile，会改坏 blanking）禁用
sudo systemctl disable rkaiq_3A
```

## 7. 设备校准（首次）

### 7.1 鱼眼参数
编辑 `web_rtc_aiortc.py` 顶部 `FISHEYE`：
```python
FISHEYE = {
    "enabled": True,
    "fx": 1000.0, "fy": 1000.0,   # 相机焦距
    "cx": 1267.0, "cy": 807.0,    # 鱼眼圆中心（检测：圆暗区中心）
    "k1": -0.08, "k2": 0.02, "k3": -0.005, "k4": 0.001,
    "fov_scale": 0.48,            # <1 视野更大
}
```
抓帧找鱼眼圆：`cv2.HoughCircles` 或边缘检测（本仓库 verify 用 `python3` 分析亮度）。

### 7.2 曝光/帧率
AE 上限 486 行（≈10ms = 50Hz 半周期）——防条纹且保帧率。若想更快：上限可调（值越小画面越暗需增益）。

### 7.3 共享内存
```bash
# /dev/shm 需可写（默认 tmpfs）
ls -ld /dev/shm   # drwxrwxrwt 即可
```

## 8. 使用

| 入口 | 功能 |
|---|---|
| `http://<IP>:8080` | 视频/检测/监控/文本聊天（免证书） |
| `https://<IP>:8443` | 同上 + 语音对话（证书"继续访问"） |

### 页面功能
- 🎥 实时视频（720p，鱼眼去畸变，YOLO 画框）
- 📊 系统面板（温度/CPU/GPU/NPU/fps 秒级）
- 💬 问设备：感知注入对话（"画面有什么？温度多少？"）
- 🎤 按住说话（HTTPS）：语音 → STT → Agent → Mac Speaker 朗读

## 9. 模型转换（如需自转 YOLO 到 RK3576）

```bash
# RK3576 是 ARMv8.0：torch 2.x aarch64 wheel 需 ARMv8.2 → Illegal instruction
# 必须 torch 1.13（老版本 aarch64 兼容 ARMv8.0）
pip3 install torch==1.13.1   # 从 pypi 下 aarch64 wheel 后本地安装

# rknn-toolkit2 2.3.2（pypi，no-deps 装 + 手动依赖）
pip3 install --no-deps rknn-toolkit2
pip3 install onnx==1.16.1 protobuf==4.25.4 numpy<=1.26.4 onnxruntime scipy

# 转换（convert.py 来自 rknn_model_zoo/examples/yolov8）
python3 convert.py yolov8n.onnx rk3576 i8
# 输出 yolo11.rknn → 重命名 yolov8n_coco.rknn
```
> YOLO26 暂不支持（rknn-toolkit2 2.3.2 op 兼容问题，等 Rockchip 更新）。
> ONNX opset>19 需先降级：`m.opset_import[0].version=19`。

## 10. 常见故障

| 症状 | 原因/解决 |
|---|---|
| 视频黑屏 | 浏览器缓存旧页 → Ctrl+Shift+R；或 SPS/PPS 缺失（mpph264enc 需 `header-mode=each-idr profile=baseline`） |
| 低帧率(<10) | vertical_blanking 被改（→54）；exposure>486 行 |
| video11 busy | camera-npu/rknn_cam 进程残留 → `fuser -k /dev/video11` |
| 语音识别差 | 麦克风距离/音量（看声压条）；whisper base 换 small（whisper-small 目录改名启用） |
| LLM 无回复 | llama-server 挂（8082 端口）；生成慢（CPU 满载 30-60s，页面等待） |
| 语音需 HTTPS | getUserMedia 安全限制；用 :8443 端口 |
| 旧服务残留 | 多次重启后 `ps aux | grep web_rtc` 多进程 → kill -9 全清 + `fuser -k /dev/video11` |
| HTTPS 无法访问 | 自签证书：浏览器"高级→继续访问"；手机浏览器建议用 HTTP |

## 11. 性能基准（RK3576 实测）

| 指标 | 值 |
|---|---|
| 视频 | 1280×720，MPP H264 硬编，15-23fps |
| YOLO 推理 | 22-27ms/次（INT8，40+FPS 能力） |
| 端到端检测 | 81-120ms/次 |
| LLM | Qwen1.5B：13 字/s（满载 3-5 字/s） |
| STT | whisper base：转录 3-4s/句 |
| 延迟 | 板端 ~120ms + 浏览器 jitter ~500ms |

## 12. 端口汇总
| 端口 | 服务 |
|---|---|
| 8080 | HTTP 页面 |
| 8443 | HTTPS 页面（语音） |
| 8081 | WebSocket 信令（HTTP 页面） |
| 8084 | WebSocket 信令（HTTPS 页面，WSS） |
| 8082 | llama-server（127.0.0.1） |

## 13. 扩展
- **分辨率**：改 `web_rtc_aiortc.py` `OUT_W/OUT_H`（720p→960×540 等）
- **YOLO 模型**：COCO 80 类 YOLO11n 已转换（yolo11n_coco.rknn）；换模型需重转
- **LLM**：Qwen2.5-1.5B 已备（qwen2.5-1.5b-q4.gguf 在板）；0.5B 更快（质量低）
- **语音**：whisper base/small 已备（small 中文更好，改服务模型路径启用）
