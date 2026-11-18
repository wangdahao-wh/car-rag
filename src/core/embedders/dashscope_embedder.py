"""DashScope API 向量化器：调用阿里百炼 OpenAI 兼容端点。

配置（.env，仅 EMBED_MODEL=dashscope 时生效）：
    DASHSCOPE_API_KEY=...            # 复用项目已有 key
    DASHSCOPE_EMBED_MODEL=text-embedding-v4
    DASHSCOPE_EMBED_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
    DASHSCOPE_EMBED_DIM=1024         # 可选，仅 v4 支持自定义维度

注意事项：
    1. 单请求最多 25 条文本（API 限制，代码已分批）
    2. v 系列服务端已封装查询指令，两侧都直接传原文，无需手工拼前缀
    3. 换模型或换维度后必须重建索引（向量空间不兼容）
"""
import logging
import os
import time
from pathlib import Path

import requests

from .base import Embedder

logger = logging.getLogger(__name__)

BATCH_SIZE = 25    # DashScope 单请求文本数上限
MAX_RETRIES = 3    # 限流/服务端错误的重试次数


class DashScopeEmbedder(Embedder):
    """DashScope API 实现：换模型只需改 DASHSCOPE_EMBED_MODEL 字符串。"""

    def __init__(self):
        self.model = os.environ.get("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")
        self.name = f"dashscope-{self.model}"
        self.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.url = os.environ.get(
            "DASHSCOPE_EMBED_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
        )
        # 可选维度（仅 v4 支持；None 表示用模型默认维度）
        dim = os.environ.get("DASHSCOPE_EMBED_DIM")
        self.dimensions = int(dim) if dim else None

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用一次 API；限流/5xx 指数退避后重试。"""
        payload = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                res = requests.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=60,
                )
                res.raise_for_status()
                data = res.json()["data"]
                # 按 index 排序，保证返回顺序与输入一致
                return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"DashScope 向量化失败（已重试 {MAX_RETRIES} 次）: {e}") from e
                wait = 2 ** attempt
                logger.warning("DashScope 请求失败（%s），%d 秒后重试 %d/%d", e, wait, attempt, MAX_RETRIES)
                time.sleep(wait)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """按 25 条一批分批向量化。"""
        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[i:i + BATCH_SIZE]))
            logger.info("向量化进度: %d/%d", min(i + BATCH_SIZE, len(texts)), len(texts))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """单条查询向量化。"""
        return self._embed_batch([query])[0]
