"""
RAG服务： 封装检索+重排+生成

关键：模型在【构造时】加载一次，之后所有请求复用同一个实例。
"""

import os
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from src.core.generate import generate, generate_stream

class RAGService:
    """端到端 RAG 服务(进程内单例)"""
    def __init__(self, index_dir: Path, collection: str = "bmw_x3_manual"):
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

        client = chromadb.PersistentClient(path=str(index_dir))
        col = client.get_collection(collection)
        vector_store = ChromaVectorStore(chroma_collection=col)
        index = VectorStoreIndex.from_vector_store(vector_store)

        self.retriever = index.as_retriever(similarity_top_k=20)
        self.reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-v2-m3", top_n=5)

    def ask(self, question: str) -> dict:
        """完整回答： 检索 -- 重排 -- 生成"""
        nodes = self.retriever.retrieve(question)
        nodes = self.reranker.postprocess_nodes(nodes,query_str=question)

        answer = generate(question,nodes)

        return {
            "answer":answer,
            "sources":[
                {
                    "page" : n.node.metadata["page"],
                    "chapter" : n.node.metadata.get("chapter", ""),
                    "section" : n.node.metadata.get("section", ""),
                    "text" : n.node.text[:100],
                }
                for n in nodes
            ],
            
        }

    def ask_stream(self, question: str):
        """流式问答： yield SSE 格式的字符串"""
        import json

        nodes = self.retriever.retrieve(question)
        nodes = self.reranker.postprocess_nodes(nodes,query_str=question)

        sources = [
            {
                "page" : n.node.metadata["page"],
                "chapter" : n.node.metadata.get("chapter", ""),
                "section" : n.node.metadata.get("section", ""),
            }
            for n in nodes
        ]
        # ② 先把【引用来源】发出去（用户在等答案时就能看到出处）
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        for token in generate_stream(question, nodes):
            yield f"data: {json.dumps({'type':'token', 'text': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"