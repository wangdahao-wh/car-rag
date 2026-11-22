"""Elasticsearch 向量库：dense_vector + HNSW + 余弦相似度检索。

不用 LlamaIndex 自带的 ElasticsearchStore：它内部是 AsyncElasticsearch，
Python 3.14 下 aiohttp 超时必须在 Task 里，入库会直接失败。
这里用同步 Elasticsearch 客户端写入和检索。

ES 是外部服务，persist_dir 只作实验组合标识：
    data/index/bge-m3-elasticsearch/chunks
    → car-rag-bge-m3-elasticsearch-chunks

连接配置（.env）：
    ES_URL=http://localhost:9200
    ES_API_KEY=...                 # 云端才填
    ES_USERNAME=elastic            # 可选
    ES_PASSWORD=...
"""
import logging
import os
import re
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.helpers.vectorstore import DenseVectorStrategy, DistanceMetric, VectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import PrivateAttr
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict
from llama_index.vector_stores.elasticsearch.utils import get_user_agent

from ..embedders import Embedder
from .base import VectorStoreBase, to_llama_embedding

logger = logging.getLogger(__name__)

TEXT_FIELD = "content"
VECTOR_FIELD = "embedding"
RRF_K = 60  # RRF 平滑常数，ES 默认值
CANDIDATE_MULT = 4  # 两路各取 4k 候选再融合：实测粗排 Hit@10 最优（2k 会丢单路靠后的金标）


def _es_field(key: str, value) -> str:
    """字符串走 keyword 精确过滤，数值走原字段。"""
    sample = value[0] if isinstance(value, list) and value else value
    if isinstance(sample, str):
        return f"metadata.{key}.keyword"
    return f"metadata.{key}"


def _to_es_filter(filters: MetadataFilters | None) -> dict | None:
    """MetadataFilters → ES bool filter。支持 EQ / NE / IN。"""
    if filters is None or not filters.filters:
        return None
    must = []
    must_not = []
    for f in filters.filters:
        field = _es_field(f.key, f.value)
        if f.operator == FilterOperator.EQ:
            must.append({"term": {field: f.value}})
        elif f.operator == FilterOperator.NE:
            must_not.append({"term": {field: f.value}})
        elif f.operator == FilterOperator.IN:
            must.append({"terms": {field: f.value}})
        else:
            raise ValueError(f"ES 过滤不支持 operator={f.operator}")
    clause: dict = {}
    if must:
        clause["must"] = must
    if must_not:
        clause["must_not"] = must_not
    return {"bool": clause} if clause else None


def _hit_to_node(hit: dict) -> BaseNode:
    """ES hit → TextNode。"""
    source = hit.get("_source", {})
    metadata = source.get("metadata", {})
    text = source.get(TEXT_FIELD)
    try:
        node = metadata_dict_to_node(metadata)
        node.text = text
        return node
    except Exception:
        return TextNode(text=text, metadata=metadata, id_=hit.get("_id"))


def _hits_to_result(hits: list[dict]) -> VectorStoreQueryResult:
    """ES hit 列表 → LlamaIndex 检索结果。"""
    return VectorStoreQueryResult(
        nodes=[_hit_to_node(h) for h in hits],
        ids=[h["_id"] for h in hits],
        similarities=[h.get("_score") for h in hits],
    )


class SyncElasticsearchStore(BasePydanticVectorStore):
    """LlamaIndex 检索接口的同步 ES 实现。"""

    stores_text: bool = True
    _client: Elasticsearch = PrivateAttr()
    _index_name: str = PrivateAttr()
    _store: VectorStore = PrivateAttr()

    def __init__(self, client: Elasticsearch, index_name: str, num_dimensions: int, **kwargs):
        super().__init__(stores_text=True, **kwargs)
        self._client = client
        self._index_name = index_name
        self._store = VectorStore(
            client=client,
            index=index_name,
            retrieval_strategy=DenseVectorStrategy(distance=DistanceMetric.COSINE),
            num_dimensions=num_dimensions,
            text_field=TEXT_FIELD,
            vector_field=VECTOR_FIELD,
        )

    @property
    def client(self) -> Elasticsearch:
        return self._client

    def add(self, nodes, **kwargs) -> list[str]:
        ids, texts, vectors, metadatas = [], [], [], []
        for node in nodes:
            ids.append(node.node_id)
            texts.append(node.get_content(metadata_mode=MetadataMode.NONE) or "")
            vectors.append(node.get_embedding())
            metadatas.append(node_to_metadata_dict(node, remove_text=True, flat_metadata=False))
        return self._store.add_texts(texts=texts, vectors=vectors, metadatas=metadatas, ids=ids)

    def delete(self, ref_doc_id: str, **kwargs) -> None:
        self._client.delete_by_query(
            index=self._index_name,
            query={"term": {"metadata.ref_doc_id.keyword": ref_doc_id}},
            refresh=True,
            ignore_unavailable=True,
        )

    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        top_k = query.similarity_top_k
        cand = top_k * CANDIDATE_MULT
        es_filter = _to_es_filter(query.filters)

        # 向量召回
        knn = {
            "field": VECTOR_FIELD,
            "query_vector": query.query_embedding,
            "k": cand,
            "num_candidates": max(50, 10 * cand),
        }
        if es_filter:
            knn["filter"] = es_filter
        knn_resp = self._client.search(index=self._index_name, knn=knn, size=cand)

        # 无 query_str（枚举取回）只走向量，不做关键词融合
        if not query.query_str:
            return _hits_to_result(knn_resp["hits"]["hits"][:top_k])

        # BM25 关键词召回：ES 开源版不支持 rank.rrf，改在应用层融合
        bool_query: dict = {"must": [{"match": {TEXT_FIELD: query.query_str}}]}
        if es_filter:
            bool_query["filter"] = es_filter
        bm25_resp = self._client.search(
            index=self._index_name, query={"bool": bool_query}, size=cand
        )

        # RRF 融合：只看两路名次，抹平向量分与 BM25 分的量纲差异
        rrf: dict[str, float] = {}
        hits: dict[str, dict] = {}
        for resp in (knn_resp, bm25_resp):
            for rank, hit in enumerate(resp["hits"]["hits"], 1):
                rrf[hit["_id"]] = rrf.get(hit["_id"], 0.0) + 1.0 / (RRF_K + rank)
                hits.setdefault(hit["_id"], hit)
        ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return VectorStoreQueryResult(
            nodes=[_hit_to_node(hits[i]) for i, _ in ordered],
            ids=[i for i, _ in ordered],
            similarities=[score for _, score in ordered],
        )

    def clear(self) -> None:
        if self._client.indices.exists(index=self._index_name):
            self._client.indices.delete(index=self._index_name)


class ElasticsearchVectorStore(VectorStoreBase):
    """Elasticsearch 实现：build 重建远端索引，load 返回检索用索引。"""

    name = "elasticsearch"

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.es_url = os.environ.get("ES_URL", "http://localhost:9200")
        self.es_api_key = os.environ.get("ES_API_KEY") or None
        self.es_username = os.environ.get("ES_USERNAME") or None
        self.es_password = os.environ.get("ES_PASSWORD") or None

        combo = f"{self.persist_dir.parent.name}-{self.persist_dir.name}".lower()
        self.index_name = "car-rag-" + re.sub(r"[^a-z0-9_-]+", "-", combo)

    def _client(self) -> Elasticsearch:
        kwargs: dict = {
            "hosts": [self.es_url],
            "headers": {"user-agent": get_user_agent()},
        }
        if self.es_api_key:
            kwargs["api_key"] = self.es_api_key
        elif self.es_username and self.es_password:
            kwargs["basic_auth"] = (self.es_username, self.es_password)
        return Elasticsearch(**kwargs)

    def _vector_store(self, num_dimensions: int) -> SyncElasticsearchStore:
        return SyncElasticsearchStore(self._client(), self.index_name, num_dimensions)

    def build(self, embedder: Embedder, nodes: list[TextNode]) -> None:
        """删除同组合旧索引并重新写入。"""
        dim = len(nodes[0].embedding) if nodes else 1024
        vector_store = self._vector_store(dim)
        vector_store.clear()
        logger.info("已清空 Elasticsearch 索引: %s", self.index_name)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info(
            "写入 Elasticsearch（%d 个 node）: %s（%s）",
            len(nodes),
            self.index_name,
            self.es_url,
        )
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=to_llama_embedding(embedder),
            show_progress=True,
        )
        vector_store.client.close()

    def load(self, embedder: Embedder) -> VectorStoreIndex:
        """加载远端索引；embedder 只用于检索时的查询向量化。"""
        logger.info("加载 Elasticsearch 索引: %s（%s）", self.index_name, self.es_url)
        dim = len(embedder.embed_query("test"))
        return VectorStoreIndex.from_vector_store(
            self._vector_store(dim), embed_model=to_llama_embedding(embedder)
        )
