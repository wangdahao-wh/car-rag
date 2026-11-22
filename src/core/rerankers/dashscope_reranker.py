"""DashScope gte-rerank-v2：阿里百炼 rerank API（与 qwen embedding 同族，全 API 化）。

对比实验用：本地 bge 换成 API rerank，零显存占用，换模型只改服务端型号。
API 文档：https://help.aliyun.com/zh/model-studio/text-rerank-api
"""
import os

import requests
from llama_index.core.schema import NodeWithScore

from .base import Reranker

API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
MODEL = "gte-rerank-v2"
BATCH_LIMIT = 25  # 单请求文档数上限（DashScope rerank 服务约束）


class DashScopeReranker(Reranker):
    """调用百炼 text-rerank 服务精排，输出与本地 reranker 完全对齐。

    注意：返回的 relevance_score 是 cross-encoder 相关性分，
    与向量召回的相似度不同源，只用于本次结果内部排序。
    """

    name = "dashscope"

    def __init__(self, top_n: int = 5):
        self.top_n = top_n
        # 未配置 DASHSCOPE_API_KEY 时直接失败（与 embedders 同一策略，不做兜底）
        self.api_key = os.environ["DASHSCOPE_API_KEY"]

    def postprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        # 超过单请求上限时分批调用，合并结果后统一排序
        results: list[NodeWithScore] = []
        for start in range(0, len(nodes), BATCH_LIMIT):
            batch = nodes[start : start + BATCH_LIMIT]
            res = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": MODEL,
                    "input": {
                        "query": query_str,
                        "documents": [n.node.get_content() for n in batch],
                    },
                    "parameters": {"return_documents": False, "top_n": len(batch)},
                },
                timeout=30,
            )
            res.raise_for_status()
            for item in res.json()["output"]["results"]:
                results.append(
                    NodeWithScore(node=batch[item["index"]].node, score=item["relevance_score"])
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: self.top_n]
