#!/bin/bash
# ============================================================
# 容器入口脚本
# 1. 若 ChromaDB 索引不存在，则从 data/x3-2024-chunks.json 自动构建
# 2. 然后执行 CMD（启动 uvicorn）
# ============================================================
set -e
cd /app

INDEX_FILE="data/index/chroma.sqlite3"

if [ ! -f "$INDEX_FILE" ]; then
    echo "[entrypoint] 未检测到 ChromaDB 索引，开始从 chunks.json 构建..."
    python src/core/build_index.py
    echo "[entrypoint] 索引构建完成。"
else
    echo "[entrypoint] 检测到已有 ChromaDB 索引，跳过构建。"
fi

exec "$@"
