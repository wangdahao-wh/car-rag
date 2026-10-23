#!/bin/bash
# ============================================================
# 容器入口脚本
# 只校验索引是否可用，不在容器启动时建库。
# 索引由宿主机执行 deploy/rebuild_index.sh 离线构建。
# ============================================================
set -e
cd /app

INDEX_DIR="data/index"

if [ ! -f "$INDEX_DIR/chroma.sqlite3" ]; then
    echo "[entrypoint] 未找到索引：/app/$INDEX_DIR/chroma.sqlite3" >&2
    echo "[entrypoint] 请先在宿主机执行：bash deploy/rebuild_index.sh" >&2
    exit 1
fi

if ! python -c 'import chromadb,sys; c=chromadb.PersistentClient(path="data/index"); col=c.get_collection("bmw_x3_manual"); sys.exit(0 if col.count() else 1)' 2>/dev/null; then
    echo "[entrypoint] 索引文件存在，但 collection 不可用或为空。" >&2
    echo "[entrypoint] 请重新执行：bash deploy/rebuild_index.sh" >&2
    exit 1
fi

echo "[entrypoint] 索引可用，启动服务。"
exec "$@"
