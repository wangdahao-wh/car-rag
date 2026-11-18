"""向量化器工厂：按环境变量 EMBED_MODEL 选择实现。

配置（.env）：
    EMBED_MODEL=bge-m3       # 本地 bge-m3（默认，无网络依赖，约 2.2GB）
    EMBED_MODEL=dashscope    # 阿里百炼 API（换模型只改 DASHSCOPE_EMBED_MODEL 字符串）

与 loaders/ 的 get_loader() 同构：读 .env → 返回统一接口的实现实例。
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import Embedder, LlamaEmbeddingAdapter

# 加载项目根目录 .env（EMBED_MODEL / DASHSCOPE_* 配置）
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

logger = logging.getLogger(__name__)


def get_embedder(name: str | None = None) -> Embedder:
    """按名称取向量化器；不传则读环境变量 EMBED_MODEL，默认 bge-m3。"""
    name = name or os.environ.get("EMBED_MODEL", "bge-m3")

    if name == "bge-m3":
        from .bge_embedder import BGEEmbedder
        return BGEEmbedder()
    elif name == "dashscope":
        from .dashscope_embedder import DashScopeEmbedder
        return DashScopeEmbedder()
    else:
        raise ValueError(f"未知的 EMBED_MODEL: {name}（可选：bge-m3 / dashscope）")


__all__ = ["Embedder", "LlamaEmbeddingAdapter", "get_embedder"]
