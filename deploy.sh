#!/bin/bash
# ============================================================
# RK3576 部署同步脚本（在 mac 端运行）
# 用法:
#   ./deploy.sh webrtc   # 同步 WebRTC 服务文件并重启（默认）
#   ./deploy.sh all      # 全部：WebRTC + 编译 rknn_cam/soft_ae
# ============================================================
set -e

BOARD="rk3576"                       # ~/.ssh/config 主机别名
REMOTE_DIR="/usr/local/bin/webrtc"
SUDO="echo myir | sudo -S"           # 板子 sudo 密码

deploy_webrtc() {
    echo "=== [1/3] 同步 WebRTC 服务文件 → 板子 ==="
    scp webrtc/web_rtc_aiortc.py webrtc/v4l2_cap.py webrtc/index.html "$BOARD:/tmp/"
    ssh "$BOARD" "$SUDO cp /tmp/web_rtc_aiortc.py /tmp/v4l2_cap.py /tmp/index.html $REMOTE_DIR/ && \
        $SUDO systemctl restart web-rtc.service"

    echo "=== [2/3] 等待服务启动 ==="
    sleep 4

    echo "=== [3/3] 验证 ==="
    ssh "$BOARD" "systemctl is-active web-rtc.service"
    ports=$(ssh "$BOARD" "ss -tlnp 2>/dev/null | grep -cE '8080|8081'")
    echo "端口监听: $ports/2"
    if [ "$ports" = "2" ]; then
        echo "✅ 部署完成! 浏览器访问 http://192.168.2.100:8080"
    else
        echo "⚠️ 端口异常，请检查 journalctl -u web-rtc.service"
    fi
}

deploy_npu() {
    echo "=== 同步并编译 rknn_cam（NPU 推理）==="
    scp rknn_cam.c "$BOARD:/tmp/"
    ssh "$BOARD" "gcc -O2 -o /tmp/rknn_cam /tmp/rknn_cam.c -lrknnrt -I/usr/include -pthread -lm && \
        $SUDO cp /tmp/rknn_cam /usr/local/bin/rknn_cam && echo '✅ rknn_cam OK'"

    echo "=== 同步并编译 soft_ae（软件曝光）==="
    scp soft_ae.c "$BOARD:/tmp/"
    ssh "$BOARD" "gcc -O2 -o /tmp/soft_ae /tmp/soft_ae.c -pthread && \
        $SUDO cp /tmp/soft_ae /usr/local/bin/soft_ae && echo '✅ soft_ae OK'"
}

case "${1:-webrtc}" in
    webrtc) deploy_webrtc ;;
    all)    deploy_webrtc && deploy_npu ;;
    *) echo "用法: $0 [webrtc|all]"; exit 1 ;;
esac
