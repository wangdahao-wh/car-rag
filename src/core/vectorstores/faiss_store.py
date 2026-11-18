"""FAISS 向量库：预留工程位（未实现）。

对比实验结论（2026-11）：向量库对检索质量影响约 1/10——本项目 4600 chunk 规模下，
Chroma 的 HNSW 与 FAISS 精确检索的 top-k 几乎逐条一致。向量库选型看工程属性
（规模/运维），不看检索质量，故降级为预留位。

未来启用步骤：
    1. pip install faiss-cpu（arm64 有官方 wheel）
    2. build：IndexFlatIP + 向量 L2 归一化（归一化后内积 = 余弦），
       storage_context.persist() 存 docstore + faiss.write_index() 存向量
    3. load：FaissVectorStore.from_persist_dir(persist_dir)
"""
import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from ..embedders import Embedder
from .base import VectorStoreBase

logger = logging.getLogger(__name__)


class FaissStore(VectorStoreBase):
    """FAISS 预留位：接口已定义，实现待需要时填充。"""

    name = "faiss"

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)

    def build(self, embedder: Embedder, nodes: list[TextNode]) -> None:
        raise NotImplementedError(
            "FAISS 为预留工程位，未实现。启用步骤见本文件模块注释"
            "（pip install faiss-cpu 后填充 IndexFlatIP + 归一化逻辑）"
        )

    def load(self, embedder: Embedder) -> VectorStoreIndex:
        raise NotImplementedError("FAISS 为预留工程位，未实现。启用步骤见本文件模块注释")
