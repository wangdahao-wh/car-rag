"""向量库抽象基类：定义建库/检索的统一协议。

所有向量库（chroma / faiss 预留）都实现 build / load，
上游（build_index 入库 pipeline）和检索侧（test_retrieval / rag_service）共用同一接口。

设计约定：
    build 之前调用方已把向量填进 node.embedding（由 Embedder 批量生成），
    向量库只负责"存储向量 + 最近邻检索"，不做向量化。
"""
from abc import ABC, abstractmethod
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from ..embedders import Embedder, LlamaEmbeddingAdapter


class VectorStoreBase(ABC):
    """向量库接口。

    统一行为：
        name                      库标识（用于索引目录命名，如 "chroma" / "faiss"）
        build(embedder, nodes)    向量化结果写入向量库并持久化（入库 pipeline）
        load(embedder)            加载为 llama_index 索引（检索 pipeline）
    """

    name: str
    persist_dir: Path

    @abstractmethod
    def build(self, embedder: Embedder, nodes: list[TextNode]) -> None:
        """把带向量的 nodes 写入向量库并持久化。"""
        raise NotImplementedError

    @abstractmethod
    def load(self, embedder: Embedder) -> VectorStoreIndex:
        """加载已持久化的索引；embedder 只用于查询侧向量化。"""
        raise NotImplementedError


def to_llama_embedding(embedder: Embedder) -> LlamaEmbeddingAdapter:
    """把 Embedder 包成 llama_index 可用的 embedding（检索侧查询向量化）。"""
    return LlamaEmbeddingAdapter(embedder)
