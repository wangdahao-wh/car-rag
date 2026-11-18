"""检索测试：加载当前配置的索引，跑测试问题，对比召回和 rerank 结果。

与 build_index 共用同一套工厂（embedder / vector store）和索引目录规则，
切 .env 换组合后，建库和检索自动指向同一份索引。

用法：
    python src/core/test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank as SbertRerank

from src.core.build_index import get_index_dir
from src.core.embedders import get_embedder
from src.core.query_filter import build_query_filters, detect_company, detect_model
from src.core.vectorstores import get_vector_store, vector_store_name

# 测试问题（问题, 期望页码）——期望页来自 BMW X3 / 领克03 用户手册
# 对比实验：同一问题【不带品牌词】vs【带品牌词】成对出现，观察品牌过滤对召回的影响
#   不带品牌词 → 全库检索（领克/BMW 同台竞争，考验 embedding + rerank 排序）
#   带品牌词   → Chroma 前置过滤（只检索目标品牌，结果应全部为对应 [company]）
TEST_QUERIES = [
    # BMW 侧：期望页来自 BMW X3 用户手册
    ("胎压数据在哪看？",                    284),
    ("宝马X3 胎压数据在哪看？",              284),
    # 领克侧：期望页来自领克 03 用户手册（p62 胎压监测系统 / p161 自适应巡航 ACC）
    ("胎压监测系统是怎么工作的？",            62),
    ("领克03 胎压监测系统是怎么工作的？",      62),
    ("自适应巡航在什么车速范围内可用？",       161),
    ("领克03 自适应巡航在什么车速范围内可用？", 161),
]

# 阶段 1：按当前配置加载索引（模型/向量库/chunk 变体由 .env 决定）
embedder = get_embedder()
index_dir = get_index_dir(embedder.name, vector_store_name())
store = get_vector_store(index_dir)
index = store.load(embedder)
print(f"索引目录: {index_dir}\n")

# 阶段 2：检索 + rerank（retriever 按问题动态创建，因品牌过滤条件而异）
rerank = SbertRerank(model="BAAI/bge-reranker-v2-m3", top_n=5)


def show_nodes(nodes, query, expect_page):
    """打印一组检索结果（命中期望页标记 ✅）。"""
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        hit = "✅" if expect_page is not None and m['page'] == expect_page else ""
        print(f"  {hit} # {i} 分数:{n.score:.4f} 第{m['page']}页 [{m.get('company', '?')}/{m.get('model', '?')}] [{m['chapter']} > {m['section']}]")
        print(f"      {n.node.text[:50].replace(chr(10), ' ')}")


for query, expect_page in TEST_QUERIES:
    model = detect_model(query)
    company = detect_company(query)
    route = f"{company}/{model}" if model else (company or "未识别（全库检索）")
    print(f"\n{'=' * 60}")
    print(f"问：{query}")
    print(f"路由: {route} | 期望页：{expect_page}")

    print("\n向量召回:")
    retriever = index.as_retriever(similarity_top_k=10, filters=build_query_filters(query))
    nodes = retriever.retrieve(query)
    show_nodes(nodes, query, expect_page)

    print("\nrerank 结果:")
    nodes = rerank.postprocess_nodes(nodes, query_str=query)
    show_nodes(nodes, query, expect_page)
