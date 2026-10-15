import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank as SbertRerank

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT / "data" / "index"
COLLECTION = "bmw_x3_manual"

# 测试问题
TEST_QUERIES = [ # test queries
    ("胎压数据在哪看？",              284),
    ("多久检查一次轮胎压力？",         284),
    ("雾天自动大灯能识别吗？",         155),
    ("怎么确认油箱加满了？",           282),
]

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

client = chromadb.PersistentClient(path=str(INDEX_DIR))
collection = client.get_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=collection)
index =  VectorStoreIndex.from_vector_store(vector_store)

retriever = index.as_retriever(similarity_top_k=10)
rerank = SbertRerank(model="BAAI/bge-reranker-v2-m3", top_n=5)

for query, expect_page in TEST_QUERIES:
    print(f"\n{'=' * 60}")
    print(f"问：{query}")
    print(f"期望：{expect_page}")

    nodes = retriever.retrieve(query)
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        hit = "✅" if m['page'] == expect_page else ""
        print(f"  {hit} # {i} 分数:{n.score:.4f} 第{m['page']}页 [{m['chapter']} > {m['section']}]")
        print(f"      {n.node.text[:50].replace(chr(10), ' ')}")

    print(f"\nrerank结果")
    nodes = rerank.postprocess_nodes(nodes, query_str=query)
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        hit = "✅" if m['page'] == expect_page else ""
        print(f"  {hit} # {i} 分数:{n.score:.4f} 第{m['page']}页 [{m['chapter']} > {m['section']}]")
        print(f"      {n.node.text[:50].replace(chr(10), ' ')}")






