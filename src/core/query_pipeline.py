"""查询流水线：加载索引、路由、对比改写、召回。

evaluate.py / test_retrieval.py / test_refuse.py 共用同一套阶段，
避免三处各自拼 retriever。
"""
import json
from pathlib import Path

from src.core.build_index import get_index_dir
from src.core.embedders import get_embedder
from src.core.generate import rewrite_compare_query
from src.core.query_filter import detect_companies, detect_models, route_query
from src.core.rerankers import get_reranker
from src.core.vectorstores import get_vector_store, vector_store_name

ROOT = Path(__file__).resolve().parent.parent.parent
QA_PATH = ROOT / "eval" / "qa_set.json"

COARSE_K = 10
FINE_K = 5
ENUM_K = 200


def load_qa(*, smoke: bool | None = None, intents: set[str] | None = None) -> list[dict]:
    """阶段：读 qa_set.json。smoke=True 只取冒烟题；intents 过滤意图。"""
    rows = [json.loads(l) for l in QA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if smoke is not None:
        rows = [r for r in rows if bool(r.get("smoke")) == smoke]
    if intents is not None:
        rows = [r for r in rows if r["intent"] in intents]
    return rows


def load_runtime(fine_k: int = FINE_K):
    """阶段：按当前 .env 加载索引和 reranker。"""
    embedder = get_embedder()
    index_dir = get_index_dir(embedder.name, vector_store_name())
    index = get_vector_store(index_dir).load(embedder)
    rerank = get_reranker(top_n=fine_k)
    return index, rerank, index_dir


def expand_compare(query: str) -> list[dict]:
    """阶段：对比题改写成单车型子问题，每条带自己的路由。"""
    rewritten = rewrite_compare_query(query, detect_models(query), detect_companies(query))
    items = []
    for sub in rewritten:
        items.append({
            "query": sub,
            "route": route_query(sub),
            "models": detect_models(sub),
        })
    return items


def match_sub(models: list[str], subs: list[dict]) -> dict | None:
    """子问题车型对上金标 subs 中的一条。"""
    for spec in subs:
        if spec["model"] in models:
            return spec
    return None


def retrieve_enumerate(index, query: str, filters) -> list:
    """枚举召回：过滤后全量取回，不走向量排序。"""
    return index.as_retriever(similarity_top_k=ENUM_K, filters=filters).retrieve(query)


def retrieve_point(index, rerank, query: str, filters) -> tuple[list, list]:
    """点查召回：粗排 top10 + 精排 top5。"""
    coarse = index.as_retriever(similarity_top_k=COARSE_K, filters=filters).retrieve(query)
    fine = rerank.postprocess_nodes(coarse, query_str=query)
    return coarse, fine
