"""ChromaDB 向量库：持久化 + 余弦相似度检索。

索引结构：persist_dir 下一个固定名称的 collection，
持久化由 PersistentClient 自动落盘，无需额外保存动作。
"""
import logging
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..embedders import Embedder
from .base import VectorStoreBase, to_llama_embedding

logger = logging.getLogger(__name__)

COLLECTION_NAME = "car_rag"


class ChromaStore(VectorStoreBase):
    """ChromaDB 实现：build 写入并自动持久化，load 返回检索用索引。"""

    name = "chroma"

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)

    def _client(self) -> chromadb.PersistentClient:
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def build(self, embedder: Embedder, nodes: list[TextNode]) -> None:
        """写入向量并自动持久化（同名 collection 先删再建，支持重建索引）。"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = self._client()

        # 重建场景：先删旧 collection，避免残留旧向量
        if COLLECTION_NAME in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION_NAME)
            logger.info("已删除旧 collection: %s", COLLECTION_NAME)

        # 显式指定余弦相似度（Chroma 默认 L2，显式指定保证跨库对比时分数口径一致）
        collection = client.create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("写入 ChromaDB（%d 个 node）: %s", len(nodes), self.persist_dir)
        # node.embedding 已填好（上游 Embedder 批量生成），此处不会再向量化
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=to_llama_embedding(embedder),
            show_progress=True,
        )

    def load(self, embedder: Embedder) -> VectorStoreIndex:
        """加载持久化索引；embedder 只用于检索时的查询向量化。"""
        client = self._client()
        collection = client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(
            vector_store, embed_model=to_llama_embedding(embedder)
        )
