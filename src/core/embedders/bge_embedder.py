"""bge-m3 本地向量化器：HuggingFaceEmbedding 加载，无网络依赖。

模型约 2.2GB，首次使用自动下载到本地缓存（~/.cache/huggingface）。
bge-m3 特性：1024 维，中英双语，查询/文档两侧都不需要指令前缀。
"""
import logging

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .base import Embedder

logger = logging.getLogger(__name__)

BGE_MODEL = "BAAI/bge-m3"


class BGEEmbedder(Embedder):
    """bge-m3 本地实现。"""

    def __init__(self):
        self.name = "bge-m3"
        logger.info("加载本地向量化模型: %s（首次使用会下载约 2.2GB）", BGE_MODEL)
        self._model = HuggingFaceEmbedding(model_name=BGE_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档（show_progress 显示进度条）。"""
        return self._model.get_text_embedding_batch(texts, show_progress=True)

    def embed_query(self, query: str) -> list[float]:
        """单条查询向量化。"""
        return self._model.get_query_embedding(query)
