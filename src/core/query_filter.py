"""查询路由：意图识别（枚举/点查/对比）+ metadata 过滤条件生成。

路由策略（单库 + metadata 前置过滤，不做物理分库）：
    1. 对比问题（多个车型或品牌）→ intent=compare，不生成过滤条件，
       由 LLM 改写成单车型问题后再分别检索
    2. 枚举问题（"智能驾驶有哪些"）→ doc_type=配置表 + category 过滤，
       全量取回行级 chunk 组装清单（top-k 截断会丢条目，枚举不走向量排序）
    3. 点查问题（"定速巡航怎么用"）→ company/model 过滤 + 向量检索 top-k

意图判定（规则版，car-agent 接入后换 LLM 版，接口不变）：
    对比 = 识别到 >1 个车型或 >1 个品牌。
    枚举 = 枚举模式命中（有什么/哪些/...）且（类别词命中 或 含"配置"）。
    两个条件缺一不可："胎压监测有什么用"有枚举词但是点查，
    "定速巡航怎么用"含类别语义但问的是用法。

车型/品牌词集合与语料文件对应（build_index.extract_company_model），
类别同义词映射与配置表原生分组名对应（table_serialserialize_table_rowsize.），
新增车型/文档时三处同步维护。
"""
import re

from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

# 品牌 → 查询中的品牌词模式（大小写不敏感）
COMPANY_PATTERNS = {
    "领克": re.compile(r"领克|lynk\s*&?\s*co", re.IGNORECASE),
    "BMW": re.compile(r"宝马|bmw|x3|m240i?|2系", re.IGNORECASE),
}

# 库外品牌指代：中文品牌名后紧跟"汽车"（如"小鹏汽车"）；非库内品牌则视为库外
_OUT_OF_SCOPE_CAR = re.compile(r"([\u4e00-\u9fa5]{2,6})汽车")


def detect_out_of_scope(query: str) -> bool:
    """用正则识别「XX汽车」形式的库外品牌指代。"""
    m = _OUT_OF_SCOPE_CAR.search(query)
    if not m:
        return False
    brand = m.group(1)
    return not any(pattern.search(brand) for pattern in COMPANY_PATTERNS.values())

# 车型 → (查询词模式, 所属品牌)；顺序敏感：长词在前（03Plus 先于 03）
# (?<!\d)03(?!\d)：前后不是数字，避免误匹配 2023 款等年份中的 03
MODEL_PATTERNS = {
    "03Plus": (re.compile(r"03\s*(?:plus|\+)", re.IGNORECASE), "领克"),
    "03":     (re.compile(r"(?<!\d)03(?!\d)"),                 "领克"),
    "X3":     (re.compile(r"x3", re.IGNORECASE),                "BMW"),
    "M240i":  (re.compile(r"m240i?", re.IGNORECASE),            "BMW"),
    "2系":    (re.compile(r"2系"),                              "BMW"),
}

# 手册车型没有独立配置表时，枚举过滤转到对应配置表车型
CONFIG_MODEL_ALIAS = {
    "2系": "M240i",
}

# 枚举问题特征：问"一组装备"而不是单个功能的细节
ENUMERATE_PATTERN = re.compile(r"有什么|有哪些|都有什么|哪些|都有|什么配置|清单")

# 标准类别 → (用户说法, 各品牌配置表的原生分组名)。
# 两家配置表的分组语义天然不对齐，不做跨品牌强行归一，
# 按语义把用户类别词映射到各品牌的一个或多个原生分组。
# 原生分组名以索引实际值为准（空值域的品牌用空列表 → 恒空结果）
CATEGORY_SYNONYMS = {
    "智能驾驶": {
        "words": ["智能驾驶", "驾驶辅助", "辅助驾驶", "辅助功能", "驾驶功能", "智驾", "adas"],
        # BMW 的主动安全/驾驶辅助在"安全性"，选装辅助包在"BMW专属定制"
        "categories": {"领克": ["智能驾控"], "BMW": ["安全性", "BMW专属定制"]},
    },
    "安全配置": {
        "words": ["安全配置", "安全性", "主动安全", "被动安全"],
        # 领克的安全装备拆在 舒适与安防（安防）和 智能驾控（主动安全）两组
        "categories": {"领克": ["舒适与安防", "智能驾控"], "BMW": ["安全性"]},
    },
    "舒适配置": {
        "words": ["舒适", "舒适配置", "座椅加热", "座椅"],
        "categories": {"领克": ["舒适与安防", "内部配置"], "BMW": ["内装"]},
    },
    "智联娱乐": {
        "words": ["娱乐", "车机", "音响", "智能互联", "智联", "中控屏"],
        # BMW 配置表无独立娱乐分组：音响/互联座舱/无线充电在"内装"，
        # 娱乐套装/高端音响选装在"BMW专属定制"（分组级映射，会带入
        # 同分组的座椅等非娱乐项，由 LLM 汇总时剔除）
        "categories": {"领克": ["智联娱乐"], "BMW": ["内装", "BMW专属定制"]},
    },
    "性能动力": {
        "words": ["性能", "动力", "参数", "技术参数", "发动机", "变速箱", "排量"],
        "categories": {"领克": ["性能参数"], "BMW": ["技术数据", "BMW技术"]},
    },
    "外观": {
        "words": ["外观", "外饰", "外形"],
        "categories": {"领克": ["外部配置"], "BMW": ["外观"]},
    },
    "内饰": {
        "words": ["内饰", "内部配置", "内饰条", "方向盘", "真皮"],
        "categories": {"领克": ["内部配置"], "BMW": ["内装", "内饰条", "座椅材质", "座椅材质*"]},
    },
    "车身颜色": {
        "words": ["外观配色", "颜色", "车漆", "车身漆", "漆"],
        "categories": {"领克": ["外部配置"], "BMW": ["车身漆"]},
    },
    "车轮": {
        "words": ["车轮", "轮毂", "轮圈", "轮胎"],
        "categories": {"领克": ["外部配置"], "BMW": ["车轮"]},
    },
    "选装定制": {
        "words": ["选装", "定制", "加装", "个性化"],
        "categories": {"领克": ["个性化选装", "运动空气套件"], "BMW": ["BMW专属定制"]},
    },
}

# 用户口语 → 手册原生术语。手册里没有「应急装备」这个词，
# 对应章节是「故障救援 > 车辆装备」（警示闪光灯/反光安全背心/警告三角标志/包扎绷带）。
# 检索时把原生术语追加在原问题后，原问题保留不动。
QUERY_SYNONYMS = {
    "应急装备": "故障救援 警示闪光灯 反光安全背心 警告三角标志 包扎绷带",
}


def expand_query(query: str) -> str:
    """把用户口语术语展开为手册原生术语，BM25 与向量检索共用。"""
    extras = [native for term, native in QUERY_SYNONYMS.items() if term in query]
    return f"{query} {' '.join(extras)}" if extras else query


def detect_models(query: str) -> list[str]:
    """识别查询中的全部车型；长词优先消耗匹配，避免 03Plus 再命中 03。"""
    found = []
    remaining = query
    for model, (pattern, _) in MODEL_PATTERNS.items():
        if pattern.search(remaining):
            found.append(model)
            remaining = pattern.sub(" ", remaining)
    return found


def detect_companies(query: str) -> list[str]:
    """识别查询中的全部品牌。"""
    return [company for company, pattern in COMPANY_PATTERNS.items() if pattern.search(query)]


def detect_model(query: str) -> str | None:
    """识别查询中的车型；无车型词返回 None。"""
    models = detect_models(query)
    return models[0] if models else None


def detect_company(query: str) -> str | None:
    """识别查询中的品牌；无品牌词返回 None。"""
    companies = detect_companies(query)
    return companies[0] if companies else None


def detect_category(query: str) -> str | None:
    """识别查询中的装备类别词，返回标准类别 key；无类别词返回 None。

    取命中的最长关键词：越长的词越具体，「外观配色」应归到车身颜色，
    不能因为「外观」先于「车身颜色」遍历就被判成外观。
    """
    best, best_len = None, 0
    for std, spec in CATEGORY_SYNONYMS.items():
        for word in spec["words"]:
            if word in query and len(word) > best_len:
                best, best_len = std, len(word)
    return best


def detect_intent(query: str) -> str:
    """意图识别：对比（compare）/ 枚举（enumerate）/ 点查（point）。"""
    if len(detect_models(query)) > 1 or len(detect_companies(query)) > 1:
        return "compare"
    if not ENUMERATE_PATTERN.search(query):
        return "point"
    if detect_category(query) or "配置" in query:
        return "enumerate"
    return "point"


def build_query_filters(query: str) -> MetadataFilters | None:
    """点查过滤条件：车型 > 品牌 > 全库。

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


def build_enumerate_filters(query: str) -> MetadataFilters | None:
    """枚举过滤条件：品牌/车型 + doc_type=配置表 + category ∈ 映射分组。

    - 类别词命中：按品牌映射到原生分组名集合（IN 匹配）
    - 该品牌没有对应分组：用不存在的哨兵值构造恒空过滤，
      让上层拿到空结果而不是退化成全表枚举
    - 无类别词（如"配置表都有什么"）：不加 category 条件，全表枚举
    """
    filters: list[MetadataFilter] = []

    model = detect_model(query)
    if model:
        _, company = MODEL_PATTERNS[model]
        brand = company
        # 2系手册没有配置表，枚举走 M240i 配置表
        config_model = CONFIG_MODEL_ALIAS.get(model, model)
        filters.append(MetadataFilter(key="company", value=company))
        filters.append(MetadataFilter(key="model", value=config_model))
    else:
        brand = detect_company(query)
        if brand:
            filters.append(MetadataFilter(key="company", value=brand))

    filters.append(MetadataFilter(key="doc_type", value="配置表"))
    # 排除非行级 chunk（整表 HTML 无 param，对枚举无用）
    filters.append(MetadataFilter(key="param", value="", operator=FilterOperator.NE))

    category = detect_category(query)
    if category:
        spec = CATEGORY_SYNONYMS[category]
        cats = set()
        for brand_key, native in spec["categories"].items():
            if brand is None or brand_key == brand:
                cats.update(native)
        # 品牌未识别时两品牌分组并集；品牌命中但无映射时为空 → 恒空结果
        filters.append(MetadataFilter(
            key="category",
            value=sorted(cats) if cats else ["(无对应分组)"],
            operator=FilterOperator.IN,
        ))

    return MetadataFilters(filters=filters) if filters else None


def route_query(query: str) -> dict:
    """查询路由总入口：意图 + 过滤条件 + 检索用问题。

    Returns:
        {"intent": "enumerate" | "point" | "compare",
         "query": 术语展开后的检索用问题,
         "filters": MetadataFilters | None,
         "category": 标准类别 key | None}
    """
    intent = detect_intent(query)
    expanded = expand_query(query)
    if intent == "compare":
        return {"intent": intent, "query": expanded, "filters": None, "category": detect_category(query)}
    if intent == "enumerate":
        return {
            "intent": intent,
            "query": expanded,
            "filters": build_enumerate_filters(query),
            "category": detect_category(query),
        }
    return {"intent": intent, "query": expanded, "filters": build_query_filters(query), "category": None}
