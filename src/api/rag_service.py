"""
RAG服务： 封装检索+重排+生成

关键：模型在【构造时】加载一次，之后所有请求复用同一个实例。
索引组合（embedding 模型 / 向量库 / chunk 变体）由 .env 决定，
与 build_index 共用同一套工厂和目录规则，保证建库/检索一致。
"""

from pathlib import Path

from src.core.build_index import get_index_dir
from src.core.embedders import get_embedder
from src.core.generate import generate, generate_stream, rewrite_compare_query
from src.core.query_filter import detect_companies, detect_models, route_query
from src.core.rerankers import get_reranker
from src.core.vectorstores import get_vector_store, vector_store_name

POINT_K = 20  # 点查/枚举的粗排召回数

class RAGService:
    """端到端 RAG 服务(进程内单例)"""
    def __init__(self):
        # 按 .env 当前配置加载索引（embedder 加载模型，store 加载持久化向量）
        embedder = get_embedder()
        index_dir = get_index_dir(embedder.name, vector_store_name())
        store = get_vector_store(index_dir)
        self.index = store.load(embedder)

        # 重排器由 .env 的 RERANK_MODEL 决定（bge / dashscope / none 对照组）
        self.reranker = get_reranker(top_n=5)

    def _retrieve(self, question: str):
        """检索 + 重排：按 route_query 的意图路由。

        - compare：拆成单车型子问题，各自路由检索后合并（否则双车型问题
          只按第一个车型过滤，另一车型的证据全被滤掉）
        - enumerate：按 route_query 给的配置表 + category 过滤
        - point：company/model 过滤 + 向量检索
        retriever 按问题动态创建，底层索引/模型是构造时加载的同一份。
        """
        route = route_query(question)
        if route["intent"] == "compare":
            return self._retrieve_compare(question)

        retriever = self.index.as_retriever(similarity_top_k=POINT_K, filters=route["filters"])
        nodes = retriever.retrieve(route["query"])
        return self.reranker.postprocess_nodes(nodes, query_str=route["query"])

    def _retrieve_compare(self, question: str):
        """对比题：LLM 改写成单车型问题，每路独立路由，按节点 id 去重合并。

        reranker 每个子问题各取 top_n，合并后节点数 = 子问题数 × top_n，
        让生成侧同时拿到两边的证据。
        """
        subs = rewrite_compare_query(question, detect_models(question), detect_companies(question))
        merged = {}
        for sub in subs:
            route = route_query(sub)
            nodes = self.index.as_retriever(
                similarity_top_k=POINT_K, filters=route["filters"]
            ).retrieve(route["query"])
            for n in self.reranker.postprocess_nodes(nodes, query_str=route["query"]):
                merged[n.node.node_id] = n
        return list(merged.values())

    def retrieve_only(self, question: str) -> dict:
        """只检索不生成（car-agent 的 clarify 歧义判定用，二期 HTTP 端点 POST /retrieve 的内核）。

        与 ask() 的区别：跳过 LLM 生成环节，只做 检索 + 重排。

        为什么需要这个方法：
            car-agent 的 clarify 节点在「无车型词问题」上，只需分析检索结果的
            车型分布与分数（是否跨 >=2 个车型且分数接近），据此判定要不要向
            用户追问车型（car-agent/docs/SDD.md 3.4 / SPEC FR-CLR-01）。
            若复用 ask()，每次判定都会白触发一次 DeepSeek 生成——
            答案被直接丢弃，费 token 且多等 2-5 秒。

        返回结构对齐 SDD 3.4 探测接口契约（sources 含 brand/model/page/score），
        agent 侧无需感知 llama_index 的 NodeWithScore 内部结构。
        注意：top_k 由构造时的 reranker.top_n 决定（当前 5），与契约默认值一致。
        """
        nodes = self._retrieve(question)
        return {
            "sources": [
                {
                    "brand": n.node.metadata.get("company", ""),
                    "model": n.node.metadata.get("model", ""),
                    "page": n.node.metadata["page"],
                    "chapter": n.node.metadata.get("chapter", ""),
                    "section": n.node.metadata.get("section", ""),
                    "score": n.score,
                    "text": n.node.text[:100],
                }
                for n in nodes
            ],
        }

    def ask(self, question: str) -> dict:
        """完整回答： 检索 -- 重排 -- 生成"""
        nodes = self._retrieve(question)

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

        nodes = self._retrieve(question)

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