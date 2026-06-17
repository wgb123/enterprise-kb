#!/usr/bin/env python3
"""Watchdog: 等阶段一 (BM25) 跑完后自动启动阶段二 (向量嵌入)。"""

import subprocess
import time
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml_path = "/home/wgb/projects/enterprise-kb/zhwiki-20260601-pages-articles-multistream.xml.bz2"
venv_python = os.path.join(project_root, ".venv", "bin", "python")
script = os.path.join(project_root, "scripts", "wiki_importer.py")

print("=" * 60)
print("📡 Watchdog 已启动")
print("   等待阶段一 (BM25) 完成后自动启动阶段二 (向量嵌入)")
print("=" * 60)

# 等待阶段一进程结束
while True:
    r = subprocess.run(
        ["pgrep", "-f", "wiki_importer.*skip-vectors"],
        capture_output=True, text=True
    )
    if not r.stdout.strip():
        print("✅ 阶段一已完成！")
        break
    pids = r.stdout.strip().split("\n")
    print(f"⏳ 阶段一运行中 (PID: {', '.join(pids)})，每 30 秒检查...")
    time.sleep(30)

# 再等 10 秒确保文件已落盘
time.sleep(10)

print("\n🚀 启动阶段二 (向量嵌入)...")
print(f"   命令: {venv_python} {script} {xml_path} --batch-size 500 --vectors-only")

os.chdir(project_root)
env = os.environ.copy()
env["HTTP_PROXY"] = "http://192.168.1.49:7890"
env["HTTPS_PROXY"] = "http://192.168.1.49:7890"
env["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"

proc = subprocess.Popen(
    ["nice", "-n", "19", "ionice", "-c", "3",
     venv_python, script, xml_path,
     "--batch-size", "500",
     "--vectors-only",
     "--max-pages", "20000"],
    env=env,
    stdout=sys.stdout,
    stderr=sys.stderr,
)

print(f"✅ 阶段二已启动 (PID: {proc.pid})")

# 等待阶段二结束
proc.wait()
print("\n✅ 阶段二已完成！")
