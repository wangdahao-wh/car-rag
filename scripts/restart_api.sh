#!/bin/bash
# 停掉占用 8000 端口的进程，然后重启服务

echo "停止旧服务。。。"
# lsof -ti :8000 | xargs kill -9 2>/dev/null
# sleep 1
pids=$(lsof -ti :8000 -sTCP:LISTEN || true)   # lsof 找不到时返回 1，|| true 兜住
if [ -n "$pids" ]; then
    kill $pids                        # 先 SIGTERM，给 3 秒收尾
fi
sleep 3
lsof -ti :8000 -sTCP:LISTEN | xargs kill -9 2>/dev/null   # 还赖着才强杀



echo "启动新服务。。。"
cd "$(dirname "$0")/.."
.venv/bin/python -m uvicorn src.api.main:app --reload --reload-dir src --port 8000