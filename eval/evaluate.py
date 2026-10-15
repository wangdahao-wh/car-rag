import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "eval" / "qa_set.json"
INDEX_PATH = ROOT / "data" / "index"
COLLECTION = "bmw_x3_manual"

RETRIEVE_N = 20
TOP_K = 5

def main():
    qa = [json.loads(l) for l in open(QA_PATH, encoding="utf-8") if l.strip()]
    qa_eval = [x for x in qa if x["groud_truth"] is not None]
    print(f"评估集 ： {len(qa_eval)} 题（另有 {len(qa) - len(qa_eval)} 题拒答测试）\n")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    collection = client.get_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=RETRIEVE_N)
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=TOP_K)

    results = []
    details = []

    for item in qa_eval:
        nodes = retriever.retrieve(item["question"])
        nodes = reranker.postprocess_nodes(nodes, query_str=item["question"])

        pages = [n.node.metadata["page"] for n in nodes]
        # print(f"** pages 的类型 {type(pages)}")
        best = 0
        for i, p in enumerate(pages, 1):
            if p in item["source_pages"]:
                best = i
                break
        results.append((item["type"],best))
        details.append((item["id"], item["question"], best))

        if item["id"] == 25:
            print(f"\n[诊断] 题{item['id']}: {item['question']}")
            print(f"  检索返回的页: {pages}")
            print(f"  gold 页: {item['source_pages']}")

    n = len(qa_eval)
    # for k in (1,3,5):
    #     hits = sum(1 for r in ranks if 0 < r <= k)
    #     print(f"Recall@{k} = {hits/n:.2f}. ({hits}/ {n})")
    # mmr = sum(1/ r if r else 0 for r in ranks) / n
    # print(f"MMR.    = {mmr:.2f}")

    print(f"\n --- 逐题明细 ---")
    for qid, q, r in details:
        mark = "✅" if r else "❌"
        print(f" {mark} [{qid:>2}] rank={r}. {q[:32]}")
    print(f"\n --- 分类统计 ---")
    for qtype, label in [("basic", "单点检索题"), ("multi_hop", "跨章节/多跳题")]:
        subset = [r for t, r in results if t == qtype]
        if not subset:
            continue
        n_sub = len(subset)
        line = f"[{label}] {n_sub} 题 ："
        for k in (1,3,5):
            hits = sum(1 for r in subset if 0 < r <= k)
            line += f"Recall@{k}={hits/n_sub:.2f}  "
        mrr_sub = sum(1/r if r else 0 for r in subset) / n_sub
        line += f"MRR={mrr_sub:.2f}"
        print("  " + line)


if __name__ == "__main__":
    main()