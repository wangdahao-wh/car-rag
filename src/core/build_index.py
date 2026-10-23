"""把chunk 向量化 并存入 ChromaDB
流程： 读 chunks.json ---> 转 Node ---> bge-m3 向量化 ---> 存入 ChromaDB
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_PATH = ROOT / "data" / "x3-2024-chunks.json"
INDEX_DIR = ROOT / "data" / "index"
COLLECTION = "bmw_x3_manual"

def load_chunks():
    """读 chunks.json """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"已加载 {len(chunks)} 个 chunk")
    return chunks

def to_node(chunks):
    """把 chunk 字典转成 LlamaIndex 的 TextNode。
    注意： ChromaDB 的 metadata 只支持【标量】（str/int/float/bool）
        不支持list/dict--所以breadcrumb 要拆成 chapter / section 两个字符串"""

    nodes = []
    for c in chunks:
        nodes.append(TextNode(
            text = c["text"],
            metadata = {
                "page":c["page"],
                "chapter":c["chapter"],
                "section":c["section"],
            },
        ))
        # bc = c["breadcrumb"]
        # nodes.append(TextNode(
        #     text = c["text"],
        #     metadata = {
        #         "page": c["page"],
        #         "chapter": bc[0] if len(bc) > 0 else "",
        #         "section": bc[1] if len(bc) > 1 else "",
        #     },
        # ))
    return nodes

def build_index(nodes):
    """向量化 + 存入 ChromaDB"""
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLLECTION)
        print(f"已删除旧的 collection: {COLLECTION}")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("开始向量化（首次会加载 bge-m3 模型，耐心等）...")
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    return index, collection


if __name__ == "__main__":
    chunks = load_chunks()
    nodes = to_node(chunks)
    index, collection = build_index(nodes)

    print(f"\n 索引完成")
    print(f"   collection 里的 node 数：{collection.count()}")
    print(f"   持久化目录 {INDEX_DIR}")
