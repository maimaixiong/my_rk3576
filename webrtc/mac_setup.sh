#!/bin/bash
# mac 端 WebRTC 接收依赖一键安装
# 用法: ./mac_setup.sh    （若 python3 是 conda/pyenv 等其他环境，请用该环境的 python3）

PYTHON="${PYTHON:-python3}"

echo "=== 检查 python 环境 ==="
$PYTHON --version
$PYTHON -c "import aiortc; print('aiortc OK', aiortc.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "=== 安装 aiortc + websockets（用户级安装）==="
    $PYTHON -m pip install --user aiortc websockets
    echo "=== 验证 ==="
    $PYTHON -c "import aiortc, websockets; print('依赖就绪')" && echo "安装成功！"
else
    echo "aiortc 已就绪，无需安装"
fi

echo ""
echo "=== 使用 ==="
echo "接收测试:  python3 mac_receiver.py 192.168.2.100 20"
echo "浏览器查看: http://192.168.2.100:8080  （无需任何依赖）"
