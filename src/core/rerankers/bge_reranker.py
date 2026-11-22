"""本地 bge-reranker-v2-m3：cross-encoder 精排（默认实现）。

本地推理，无网络依赖；M1 上约 0.3 秒/批（top-20 候选）。
"""
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from .base import Reranker

MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker(Reranker):
    """封装 llama_index 的 SentenceTransformerRerank，模型加载一次复用。"""

    name = "bge"

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        # SentenceTransformerRerank 内部加载 cross-encoder 权重（约 2.2GB，只加载一次）
        self._impl = SentenceTransformerRerank(model=MODEL, top_n=top_n)

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        return self._impl.postprocess_nodes(nodes, query_str=query_str)
