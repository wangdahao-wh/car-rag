"""重排器工厂：按环境变量 RERANK_MODEL 选择实现。

配置（.env）：
    RERANK_MODEL=bge         # 本地 bge-reranker-v2-m3（默认）
    RERANK_MODEL=dashscope   # 百炼 gte-rerank-v2 API
    RERANK_MODEL=none        # 不重排（对照组，量化 rerank 贡献）

与 embedders/ 的 get_embedder() 同构：读 .env → 返回统一接口的实现实例。
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import NoneReranker, Reranker

# 加载项目根目录 .env（RERANK_MODEL / DASHSCOPE_API_KEY）
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)


def get_reranker(name: str | None = None, top_n: int = 5) -> Reranker:
    """按名称取重排器；不传则读环境变量 RERANK_MODEL，默认 bge。"""
    name = name or os.environ.get("RERANK_MODEL", "bge")

    if name == "bge":
        from .bge_reranker import BGEReranker
        return BGEReranker(top_n=top_n)
    elif name == "dashscope":
        from .dashscope_reranker import DashScopeReranker
        return DashScopeReranker(top_n=top_n)
    elif name == "none":
        return NoneReranker(top_n=top_n)
    else:
        raise ValueError(f"未知的 RERANK_MODEL: {name}（可选：bge / dashscope / none）")


__all__ = ["Reranker", "NoneReranker", "get_reranker"]
