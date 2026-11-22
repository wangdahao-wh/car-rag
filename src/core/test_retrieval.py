"""检索冒烟流水线：从 qa_set 抽 smoke 题，人工看召回和 rerank。

正式指标在 eval/evaluate.py，金标全集在 eval/qa_set.json。
本脚本只打印样例，不算 Hit@k / Recall。

用法：
    python src/core/test_retrieval.py
"""
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.query_pipeline import (
    QA_PATH,
    expand_compare,
    load_qa,
    load_runtime,
    match_sub,
    retrieve_enumerate,
    retrieve_point,
)
from src.core.query_filter import route_query
from src.core.utils import setup_logging
from src.core.vectorstores import vector_store_name

logger = logging.getLogger(__name__)
detail_logger = logging.getLogger(f"{__name__}.detail")  # 独立 logger 用于逐题详情
detail_logger.propagate = False  # 不传播到 root，避免详情写进主日志


def show_nodes(nodes, expect_pages):
    """打印点查结果，命中期望页标 OK。"""
    pages = set(expect_pages or [])
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        hit = "OK " if m["page"] in pages else ""
        detail_logger.info(
            f"  {hit}# {i} 分数:{n.score:.4f} 第{m['page']}页 "
            f"[{m.get('company', '?')}/{m.get('model', '?')}] [{m['chapter']} > {m['section']}]"
        )
        detail_logger.info(f"      {n.node.text[:50].replace(chr(10), ' ')}")


def show_enumeration(nodes, category, expect_params=None):
    """打印枚举结果：按 category 分组，并对照期望 param 清单。"""
    by_cat = defaultdict(list)
    got = []
    for n in nodes:
        m = n.node.metadata
        param = m.get("param") or ""
        by_cat[m.get("category") or "(无分组)"].append(param)
        if param:
            got.append(param)
    detail_logger.info(f"  共 {len(nodes)} 条（标准类别: {category}）")
    for cat, params in sorted(by_cat.items()):
        head = "；".join(params[:6])
        more = f" ...等 {len(params)} 项" if len(params) > 6 else ""
        detail_logger.info(f"  [{cat}] {head}{more}")
    if expect_params is None:
        return
    missing = [p for p in expect_params if p not in got]
    extra = [p for p in got if p not in expect_params]
    detail_logger.info(f"  期望 {len(expect_params)} 项，命中 {len(expect_params) - len(missing)}，缺 {len(missing)}，多 {len(extra)}")
    if missing:
        detail_logger.info(f"  缺失: {missing[:8]}")
    if extra:
        detail_logger.info(f"  多余: {extra[:8]}")


def show_point(index, rerank, query, filters, pages):
    detail_logger.info(f"期望页: {pages}")
    coarse, fine = retrieve_point(index, rerank, query, filters)
    detail_logger.info("\n向量召回:")
    show_nodes(coarse, pages)
    detail_logger.info("\nrerank 结果:")
    show_nodes(fine, pages)


def show_enum(index, query, route, params):
    show_enumeration(retrieve_enumerate(index, query, route["filters"]), route["category"], params)


def show_compare(index, rerank, case):
    items = expand_compare(case["question"])
    detail_logger.info(f"改写: {[it['query'] for it in items]}")
    for spec in case["subs"]:
        if spec["intent"] == "enumerate":
            detail_logger.info(f"  期望 [{spec['model']}] {len(spec['params'])} 项")
        else:
            detail_logger.info(f"  期望 [{spec['model']}] {spec['answer']}")
    for item in items:
        spec = match_sub(item["models"], case["subs"])
        detail_logger.info(f"\n  子问：{item['query']}")
        detail_logger.info(f"  路由: {item['route']['intent']} | 类别: {item['route']['category']}")
        if item["route"]["intent"] == "enumerate":
            show_enum(index, item["query"], item["route"], spec.get("params") if spec else None)
        else:
            pages = spec.get("source_pages") if spec else None
            coarse, _ = retrieve_point(index, rerank, item["query"], item["route"]["filters"])
            show_nodes(coarse, pages)


def run_one(index, rerank, case: dict) -> None:
    """单题展示：路由 → 按意图打印召回。"""
    query = case["question"]
    route = route_query(query)
    intent_ok = "OK" if route["intent"] == case["intent"] else f"期望 {case['intent']}"
    detail_logger.info(f"\n{'=' * 60}")
    detail_logger.info(f"#{case['id']} 问：{query}")
    detail_logger.info(f"路由: {route['intent']} | 类别: {route['category']} | {intent_ok}")

    if route["intent"] == "compare":
        show_compare(index, rerank, case)
        return
    if route["intent"] == "enumerate":
        show_enum(index, query, route, case.get("params"))
        return
    show_point(index, rerank, query, route["filters"], case.get("source_pages"))


def main():
    from src.core.utils.logging_utils import ImmediateFileHandler
    
    log_file = setup_logging("test_retrieval", vector_store_name())
    
    # 给 detail_logger 加两个 handler：控制台 + 独立文件
    detail_file = log_file.parent / log_file.name.replace(".log", "_details.log")
    detail_handler = ImmediateFileHandler(detail_file, encoding="utf-8")
    detail_handler.setLevel(logging.INFO)
    detail_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    
    detail_logger.addHandler(detail_handler)
    detail_logger.addHandler(console_handler)
    detail_logger.setLevel(logging.INFO)
    
    # 阶段 1：加载冒烟集
    cases = load_qa(smoke=True, intents={"point", "enumerate", "compare"})

    # 阶段 2：加载索引
    index, rerank, index_dir = load_runtime()
    logger.info("索引目录: %s", index_dir)
    logger.info("冒烟 %d 题（全集 %s）", len(cases), QA_PATH)
    logger.info("逐题详情见: %s", detail_file)

    # 阶段 3：逐题展示
    for case in cases:
        run_one(index, rerank, case)
    
    logger.info("=" * 60)
    logger.info("冒烟完成，日志: %s", log_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
