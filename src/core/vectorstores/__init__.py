"""向量库工厂：按环境变量 VECTOR_STORE 选择实现。

配置（.env）：
    VECTOR_STORE=chroma    # 默认，自带持久化
    VECTOR_STORE=faiss     # 预留工程位，未实现

与 loaders/ 的 get_loader() 同构：读 .env → 返回统一接口的实现实例。

索引目录隔离约定（调用方负责拼路径）：
    data/index/{EMBED_MODEL 名}-{向量库名}/{chunk 变体名}/
    不同组合的索引互不覆盖，便于检索效果对比。
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import VectorStoreBase

# 加载项目根目录 .env（VECTOR_STORE 配置）
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)


def vector_store_name() -> str:
    """当前配置的向量库名称（用于索引目录命名）。"""
    return os.environ.get("VECTOR_STORE", "chroma")


def get_vector_store(persist_dir: Path) -> VectorStoreBase:
    """按 VECTOR_STORE 取向量库实例；persist_dir 为该组合的索引持久化目录。"""
    name = vector_store_name()

    if name == "chroma":
        from .chroma_store import ChromaStore
        return ChromaStore(persist_dir)
    elif name == "faiss":
        from .faiss_store import FaissStore
        return FaissStore(persist_dir)
    else:
        raise ValueError(f"未知的 VECTOR_STORE: {name}（可选：chroma / faiss）")


__all__ = ["VectorStoreBase", "vector_store_name", "get_vector_store"]
