"""重排器抽象基类：统一 rerank 调用协议。

所有重排器实现 postprocess_nodes(nodes, query_str) -> list[NodeWithScore]，
上游（RAGService / test_retrieval）只依赖该接口，具体实现由工厂切换。

设计约定：
    输入 nodes 已经过向量召回（分数是 embedding 相似度），
    重排器用 cross-encoder 精算"问题-文档"相关性，重打分并重排序。
    重排分数与召回分数不同源，仅用于组内排序，不能跨组比较。
"""
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore


class Reranker(ABC):
    """重排器接口。

    统一行为：
        name     标识（用于日志与实验记录，如 "bge" / "dashscope" / "none"）
        top_n    重排后保留的条数
    """

    name: str
    top_n: int

    @abstractmethod
    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        """对召回结果重打分、重排序，返回前 top_n 条。"""
        raise NotImplementedError


class NoneReranker(Reranker):
    """不重排（对照组）：保持向量召回的顺序和分数，只截断到 top_n。

    用途：跑"无 rerank"基线实验，量化 reranker 本身的贡献——
    否则只能比较"两个 reranker 谁好"，回答不了"rerank 值不值得要"。
    """

    name = "none"

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        return nodes[: self.top_n]
