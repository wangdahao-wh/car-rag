#!/bin/bash
# ============================================================
# 用与运行服务相同的镜像，跑一次性容器重建 ChromaDB 索引
# 用法：bash deploy/rebuild_index.sh
# 数据更新后执行；代码未变时 build 会走缓存，不产生新的镜像层
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f ".env" ]; then
    echo "[rebuild_index] 未找到 .env，请先执行 cp .env.example .env 并填写 DEEPSEEK_API_KEY" >&2
    exit 1
fi

if [ ! -f "data/x3-2024-chunks.json" ]; then
    echo "[rebuild_index] 未找到 data/x3-2024-chunks.json" >&2
    echo "[rebuild_index] 请先准备 chunks 数据工件，再重建索引。" >&2
    exit 1
fi

mkdir -p data/index

echo "[rebuild_index] 确保镜像为最新代码（代码未变时走 Docker 缓存）..."
docker compose build car-rag

echo "[rebuild_index] 停止应用，避免索引文件被占用..."
docker compose stop car-rag 2>/dev/null || true

echo "[rebuild_index] 用一次性容器重建索引..."
docker compose run --rm --no-deps \
    -v "$PWD/data:/app/data" \
    --entrypoint python \
    car-rag src/core/build_index.py

echo "[rebuild_index] 启动应用..."
docker compose up -d car-rag
