"""向量化器抽象基类：定义统一的输出协议。

所有向量化器（bge-m3 本地 / dashscope API）都实现 embed_documents / embed_query，
下游（向量库构建 / 检索）不关心向量是怎么生成的——与 loaders/base.py 的 PDFLoader 同构。
"""
from abc import ABC, abstractmethod
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class Embedder(ABC):
    """向量化器接口。

    统一行为：
        name                    模型标识（用于索引目录命名，如 "bge-m3" / "dashscope-text-embedding-v4"）
        embed_documents(texts)  文档侧向量化（入库时用，批量）
        embed_query(query)      查询侧向量化（检索时用，单条）
    """

    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批文档文本转成向量（顺序与输入一致）。"""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """把一条查询转成向量。"""
        raise NotImplementedError


class LlamaEmbeddingAdapter(BaseEmbedding):
    """把 Embedder 适配成 llama_index 的 BaseEmbedding。

    用途：检索侧（index.as_retriever()）内部会调用 embed_model 做查询向量化，
    本适配器把这些调用转发给统一的 Embedder 实现，保证建库/检索用同一个模型。
    只实现同步路径；异步路径直接调用同步实现（本项目无异步检索场景）。
    """

    _embedder: Any = PrivateAttr()

    def __init__(self, embedder: Embedder, **kwargs):
        super().__init__(model_name=embedder.name, **kwargs)
        self._embedder = embedder

    @classmethod
    def class_name(cls) -> str:
        return "LlamaEmbeddingAdapter"

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embedder.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embedder.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embedder.embed_documents([text])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embedder.embed_documents([text])[0]
