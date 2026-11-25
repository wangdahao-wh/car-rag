"""正式评测流水线：qa_set.json → 路由准确率 + 粗排/精排 + 枚举 Recall/Precision。

分工：
    eval/qa_set.json          金标全集
    eval/evaluate.py          本文件，汇总指标
    src/core/test_retrieval.py  只跑 smoke 子集，人工看样例
    eval/test_refuse.py       拒答

用法：
    python eval/evaluate.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.query_pipeline import (
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


def gold_rank(nodes, pages=None, model=None, answer=None) -> int | None:
    """第一条金标所在名次；未命中返回 None。

    「所在」按页码范围判定：chunk 覆盖 [page, page_end] 区间，金标页落在
    区间内即算命中。只认起始页会把跨页 section 中间页的答案判成未命中
    （如手册前言覆盖 p1~p8，答案在第 7 页）。
    """
    pages = set(pages or [])
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        if model and m.get("model") != model:
            continue
        if pages:
            start = m.get("page")
            end = m.get("page_end") or start
            if start is not None and any(start <= g <= end for g in pages):
                return i
        if answer and answer in (n.node.text or ""):
            return i
    return None


def hit_at(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def mrr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_point(coarse, fine, pages, model=None, answer=None) -> dict:
    """点查计分：粗排 vs 精排。金标不进粗排记 unrecoverable。"""
    cr = gold_rank(coarse, pages=pages, model=model, answer=answer)
    fr = gold_rank(fine, pages=pages, model=model, answer=answer)
    recoverable = cr is not None
    return {
        "coarse_rank": cr,
        "fine_rank": fr,
        "coarse_hit1": hit_at(cr, 1),
        "coarse_hit5": hit_at(cr, 5),
        "coarse_hit10": hit_at(cr, 10),
        "coarse_mrr": mrr(cr),
        "fine_hit1": hit_at(fr, 1),
        "fine_hit5": hit_at(fr, 5),
        "fine_mrr": mrr(fr),
        "unrecoverable": not recoverable,
        "gain_hit1": (hit_at(fr, 1) - hit_at(cr, 1)) if recoverable else 0.0,
        "gain_mrr": (mrr(fr) - mrr(cr)) if recoverable else 0.0,
        "rank_shift": (cr - fr) if cr and fr else 0,
    }


def score_enum(nodes, gold_params: list[str]) -> dict:
    """枚举计分：Recall / Precision。"""
    got = [n.node.metadata.get("param") or "" for n in nodes]
    got = [p for p in got if p]
    gold = list(gold_params)
    if not gold:
        empty = 1.0 if not got else 0.0
        return {"recall": empty, "precision": empty, "n_got": len(got), "n_gold": 0}
    hit = sum(1 for p in gold if p in got)
    precision = (sum(1 for p in got if p in gold) / len(got)) if got else 0.0
    return {"recall": hit / len(gold), "precision": precision, "n_got": len(got), "n_gold": len(gold)}


def eval_point(index, rerank, query, filters, pages, model=None, answer=None) -> dict:
    coarse, fine = retrieve_point(index, rerank, query, filters)
    return score_point(coarse, fine, pages, model, answer)


def eval_enum(index, query, filters, params) -> dict:
    return score_enum(retrieve_enumerate(index, query, filters), params)


def eval_compare(index, rerank, q: dict) -> tuple[list[dict], list[dict], list[str]]:
    """对比题：改写 → 子问题分别计分。"""
    points, enums = [], []
    items = expand_compare(q["question"])
    lines = [f"     改写: {[it['query'] for it in items]}"]
    for item in items:
        spec = match_sub(item["models"], q["subs"])
        if spec is None:
            lines.append(f"     子问未对上金标: {item['query']}")
            continue
        route = item["route"]
        if spec["intent"] == "enumerate":
            r = eval_enum(index, route["query"], route["filters"], spec.get("params") or [])
            enums.append(r)
            lines.append(
                f"     [{spec['model']}] Recall={r['recall']:.2f} Precision={r['precision']:.2f} "
                f"({r['n_got']}/{r['n_gold']})"
            )
        else:
            r = eval_point(
                index, rerank, route["query"], route["filters"],
                spec.get("source_pages") or [], spec.get("model"), spec.get("answer"),
            )
            points.append(r)
            miss = "未召回" if r["unrecoverable"] else ""
            lines.append(f"     [{spec['model']}] 粗排@{r['coarse_rank']} 精排@{r['fine_rank']} {miss}".rstrip())
    return points, enums, lines


def eval_one(index, rerank, q: dict) -> dict:
    """单题：路由 → 按意图检索计分。"""
    query = q["question"]
    route = route_query(query)
    intent_ok = route["intent"] == q["intent"]
    lines = [
        f"#{q['id']:02d} [{q['intent']}] {query} | 路由 {route['intent']} "
        f"{'OK' if intent_ok else '期望 ' + q['intent']}"
    ]
    points, enums = [], []

    if q["intent"] == "compare":
        p, e, extra = eval_compare(index, rerank, q)
        points, enums, lines = points + p, enums + e, lines + extra
    elif q["intent"] == "enumerate":
        r = eval_enum(index, route["query"], route["filters"], q.get("params") or [])
        enums.append(r)
        lines.append(f"     Recall={r['recall']:.2f} Precision={r['precision']:.2f} ({r['n_got']}/{r['n_gold']})")
    else:
        r = eval_point(index, rerank, route["query"], route["filters"], q.get("source_pages") or [], q.get("model"))
        points.append(r)
        status = "未召回" if r["unrecoverable"] else f"粗排@{r['coarse_rank']} → 精排@{r['fine_rank']}"
        lines.append(f"     {status}")

    return {"intent_ok": intent_ok, "points": points, "enums": enums, "lines": lines}


def summarize(results: list[dict]) -> None:
    """阶段：汇总打印。"""
    route_ok = [1.0 if r["intent_ok"] else 0.0 for r in results]
    points = [p for r in results for p in r["points"]]
    enums = [e for r in results for e in r["enums"]]

    logger.info("\n" + "=" * 60)
    logger.info(f"路由准确率: {mean(route_ok):.3f}  ({sum(route_ok):.0f}/{len(route_ok)})")
    if points:
        recoverable = [p for p in points if not p["unrecoverable"]]
        logger.info("点查:")
        logger.info(
            f"  粗排 Hit@1/5/10 = {mean([p['coarse_hit1'] for p in points]):.3f} / "
            f"{mean([p['coarse_hit5'] for p in points]):.3f} / "
            f"{mean([p['coarse_hit10'] for p in points]):.3f}  "
            f"MRR={mean([p['coarse_mrr'] for p in points]):.3f}"
        )
        logger.info(
            f"  精排 Hit@1/5    = {mean([p['fine_hit1'] for p in points]):.3f} / "
            f"{mean([p['fine_hit5'] for p in points]):.3f}  "
            f"MRR={mean([p['fine_mrr'] for p in points]):.3f}"
        )
        logger.info(
            f"  可重排样本 ΔHit@1={mean([p['gain_hit1'] for p in recoverable]):.3f} "
            f"ΔMRR={mean([p['gain_mrr'] for p in recoverable]):.3f} "
            f"平均名次前移={mean([p['rank_shift'] for p in recoverable]):.2f}"
        )
        logger.info(f"  粗排未召回（精排救不回）: {sum(p['unrecoverable'] for p in points)}/{len(points)}")
    if enums:
        logger.info("枚举:")
        logger.info(
            f"  平均 Recall={mean([e['recall'] for e in enums]):.3f}  "
            f"Precision={mean([e['precision'] for e in enums]):.3f}  n={len(enums)}"
        )


def main():
    log_file = setup_logging("evaluate", vector_store_name())
    
    # 阶段 1：加载评估集（拒答不进检索指标）
    qa = load_qa(intents={"point", "enumerate", "compare"})
    refuse_n = len(load_qa(intents={"refuse"}))
    logger.info(f"评估集：{len(qa)} 题（另有 {refuse_n} 题拒答，走 test_refuse.py）\n")

    # 阶段 2：加载索引
    index, rerank, index_dir = load_runtime()
    logger.info(f"索引目录: {index_dir}\n")

    # 阶段 3：逐题评测
    results = []
    for q in qa:
        item = eval_one(index, rerank, q)
        results.append(item)
        logger.info("\n".join(item["lines"]))

    # 阶段 4：汇总
    summarize(results)
    
    logger.info("=" * 60)
    logger.info(f"评测完成，日志: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
