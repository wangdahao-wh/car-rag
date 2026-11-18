"""PDF 加载器抽象基类：定义统一的输出协议。

所有加载器（pypdf / mineru）都实现 load()，返回统一结构的 chunk 列表，
下游（build_index / 检索 / 评估）不关心 PDF 是怎么解析的。
"""
from abc import ABC, abstractmethod
from pathlib import Path


class PDFLoader(ABC):
    """PDF 加载器接口。

    统一输出（chunk 字典）：
        {
            "text":    "正文文本（含标题路径前缀）",
            "page":    12,                  # 页码（从 1 开始；无法确定时为 None）
            "chapter": "车灯",              # 一级标题
            "section": "自动车灯控制",      # 二级标题
            "path":    ["车灯", "自动车灯控制"],  # 完整标题路径
        }
    """

    @abstractmethod
    def load(self, pdf_path: Path) -> list[dict]:
        """加载并切分一个 PDF，返回 chunk 列表。"""
        raise NotImplementedError
