"""拒答测试：手册里没有答案的问题，系统应该老实说「无法确定」"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from src.core.generate import generate

ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "eval" / "qa_set.json"
INDEX_PATH = ROOT / "data" / "index"
COLLECTION = "bmw_x3_manual"

RETRIEVE_N = 20
TOP_K = 5

REFUSE_QUESTIONS = [
    "这辆车在火星上能开吗？",
    "换个轮胎要多少钱？",
    "这车和奔驰 GLC 哪个好？",
]

def main():
    qa = [json.loads(l) for l in open(QA_PATH, encoding="utf-8") if l.strip()]
    qa_eval = [x for x in qa if x["groud_truth"] is not None]
    print(f"评估集 ： {len(qa_eval)} 题（另有 {len(qa) - len(qa_eval)} 题拒答测试）\n")

    # ① 全局配置：指定用哪个 embedding 模型
    #    必须在任何检索/向量化之前设置，因为 LlamaIndex 内部会读这个全局配置
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    # ② 连接本地持久化的 ChromaDB（读磁盘上的 data/index 目录）
    client = chromadb.PersistentClient(path=str(INDEX_PATH))

    # ③ 取出【已经建好的】集合（不新建——集合是 build_index.py 建好的）
    collection = client.get_collection(COLLECTION)

    # ④ 把 Chroma 的 collection 包装成 LlamaIndex 认识的向量库接口
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # ⑤ 从已有向量库【加载】索引（不重新向量化，秒级完成）
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=RETRIEVE_N)
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=TOP_K)

    for q in REFUSE_QUESTIONS:
        nodes = retriever.retrieve(q)
        nodes = reranker.postprocess_nodes(nodes, query_str=q)
        answer = generate(q, nodes[:3]) # 只取前3个上下文

        print(f"=== 问：{q} ===")
        print(f"  检索到的页：{[n.node.metadata['page'] for n in nodes[:3]]}")
        print(f"  答案： {answer[:200]}")
        print()

if __name__ == "__main__":
    main()