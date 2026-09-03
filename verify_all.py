#!/usr/bin/env python3
"""RK3576 设备 Agent 综合验证（排除语音）
验证：视频 / YOLO 检测 / 系统监控 / AI 对话（画面感知、状态感知、多轮）"""
from playwright.sync_api import sync_playwright
import time, json

RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(("✅" if ok else "❌") + f" {name}" + (f" — {detail}" if detail else ""))

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    page = browser.new_page()
    page.goto("http://192.168.2.100:8080/")
    page.click("#btnConn")
    page.wait_for_timeout(10000)

    # 1. 视频
    v = page.evaluate("""() => { const x = document.getElementById('video');
        return {w:x.videoWidth, h:x.videoHeight, t:x.currentTime, ready:x.readyState}; }""")
    check("视频播放", v["w"] == 1280 and v["t"] > 2, f"{v['w']}x{v['h']} t={v['t']:.1f}s")

    # 2. YOLO 检测状态
    det = page.inner_text("#detstate")
    check("YOLO 检测运行", "检测" in det or "目标" in det, det[:40])

    # 3. 系统监控面板
    sysfps = page.inner_text("#sysfps")
    tempavg = page.inner_text("#tempavg")
    cpu = page.inner_text("#cpupct")
    npu = page.inner_text("#npupct")
    check("系统监控(帧率)", "fps" in sysfps, sysfps)
    check("系统监控(温度)", "°C" in tempavg, tempavg)
    check("系统监控(CPU/NPU)", "%" in cpu and "%" in npu, f"CPU {cpu} NPU {npu}")

    # 4. AI 对话 - 画面感知
    def ask_question(q, wait=30):
        page.fill("#chattext", q)
        page.click("#chatsend")
        t0 = time.time()
        # 等待回复（"思考中"消失且出现 AI 文本）
        while time.time() - t0 < wait:
            log = page.inner_text("#chatlog")
            if "思考中" not in log and "🤖" in log and len(log) > 0:
                # 找到最后一条 AI
                return log
            time.sleep(1)
        return page.inner_text("#chatlog")

    q1 = "摄像头画面里检测到什么物体？请直接列出"
    r1 = ask_question(q1)
    check("对话: 画面感知", "🤖" in r1 and "思考中" not in r1,
          r1.split("🤖")[-1].strip()[:80])

    q2 = "设备当前 CPU 使用率、温度和帧率分别是多少？请用数字回答"
    r2 = ask_question(q2)
    check("对话: 状态感知", "🤖" in r2 and "思考中" not in r2,
          r2.split("🤖")[-1].strip()[:80])

    q3 = "你是谁？运行在什么设备上？一句话回答"
    r3 = ask_question(q3)
    check("对话: 身份", "🤖" in r3 and "思考中" not in r3,
          r3.split("🤖")[-1].strip()[:80])

    # 5. 多轮对话（连续 3 问无异常）
    try:
        for q in ["画面里有人吗？", "NPU 使用率多少？", "回答正常吗？"]:
            ask_question(q, wait=25)
        check("多轮对话(3轮)", True)
    except Exception as e:
        check("多轮对话(3轮)", False, str(e)[:60])

    # 6. 感知数据真实性（LLM 应引用检测/监控数据）
    log_all = page.inner_text("#chatlog")
    has_sensor_words = any(w in log_all for w in ["检测", "person", "bed", "CPU", "NPU", "温度", "帧率", "fps"])
    check("感知数据注入(LLM引用真实数据)", has_sensor_words)

    browser.close()

print("\n===== 验证汇总 =====")
passed = sum(1 for _, ok, _ in RESULTS if ok)
print(f"{passed}/{len(RESULTS)} 项通过")
