"""查询路由：识别问题中的车型/品牌词，生成 Chroma metadata 过滤条件。

路由策略（单库 + metadata 前置过滤，不做物理分库）：
    1. 识别到车型词 → company + model 双条件过滤（精度最高）
    2. 只识别到品牌词 → company 过滤（跨车型召回）
    3. 都没有 → 不过滤，全库检索（合法语义：用户未限定品牌）

过滤条件会下推到 Chroma（where 前置过滤，先筛再搜），保证 top-k
全部来自目标车型/品牌。领克/宝马手册词汇分布高度重叠，跨品牌 chunk
是稳定的强竞争噪音，路由过滤能显著提升带品牌词查询的召回质量。

车型/品牌词集合与语料文件对应（build_index.extract_company_model），
新增车型时两处同步维护。
"""
import re

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

# 品牌 → 查询中的品牌词模式（大小写不敏感）
COMPANY_PATTERNS = {
    "领克": re.compile(r"领克|lynk\s*&?\s*co", re.IGNORECASE),
    "BMW": re.compile(r"宝马|bmw|x3|m240i|2系", re.IGNORECASE),
}

# 车型 → (查询词模式, 所属品牌)；顺序敏感：长词在前（03Plus 先于 03）
# (?<!\d)03(?!\d)：前后不是数字，避免误匹配 2023 款等年份中的 03
MODEL_PATTERNS = {
    "03Plus": (re.compile(r"03\s*plus", re.IGNORECASE), "领克"),
    "03":     (re.compile(r"(?<!\d)03(?!\d)"),          "领克"),
    "X3":     (re.compile(r"x3", re.IGNORECASE),         "BMW"),
    "M240i":  (re.compile(r"m240i", re.IGNORECASE),      "BMW"),
    "2系":    (re.compile(r"2系"),                       "BMW"),
}


def detect_model(query: str) -> str | None:
    """识别查询中的车型；无车型词返回 None。"""
    for model, (pattern, _) in MODEL_PATTERNS.items():
        if pattern.search(query):
            return model
    return None


def detect_company(query: str) -> str | None:
    """识别查询中的品牌；无品牌词返回 None。"""
    for company, pattern in COMPANY_PATTERNS.items():
        if pattern.search(query):
            return company
    return None


def build_query_filters(query: str) -> MetadataFilters | None:
    """按查询生成路由过滤条件：车型 > 品牌 > 全库。

    - 识别到车型：company + model 双条件（车型属于唯一品牌，双保险）
    - 只识别到品牌：仅 company 条件
    - 都没有：返回 None（不过滤）
    """
    model = detect_model(query)
    if model:
        _, company = MODEL_PATTERNS[model]
        return MetadataFilters(filters=[
            MetadataFilter(key="company", value=company),
            MetadataFilter(key="model", value=model),
        ])

    company = detect_company(query)
    if company:
        return MetadataFilters(filters=[MetadataFilter(key="company", value=company)])

    return None
