"""拒答流水线：qa_set 里 intent=refuse 的题，系统应回答无法确定。

检索评测见 evaluate.py。
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.generate import generate
from src.core.query_filter import route_query
from src.core.query_pipeline import load_qa, load_runtime, retrieve_point
from src.core.utils import setup_logging
from src.core.vectorstores import vector_store_name

logger = logging.getLogger(__name__)


def run_one(index, rerank, q: dict) -> None:
    """单题：点查召回 → 生成答案。"""
    query = q["question"]
    route = route_query(query)
    _, fine = retrieve_point(index, rerank, query, route["filters"])
    answer = generate(query, fine[:3])
    logger.info(f"=== 问：{query} ===")
    logger.info(f"  检索到的页：{[n.node.metadata['page'] for n in fine[:3]]}")
    logger.info(f"  答案： {answer[:200]}")
    logger.info("")


def main():
    log_file = setup_logging("test_refuse", vector_store_name())
    
    # 阶段 1：加载拒答集
    refuse = load_qa(intents={"refuse"})
    logger.info(f"拒答题：{len(refuse)}（检索评测见 evaluate.py）\n")

    # 阶段 2：加载索引
    index, rerank, _ = load_runtime()

    # 阶段 3：逐题生成
    for q in refuse:
        run_one(index, rerank, q)
    
    logger.info("=" * 60)
    logger.info(f"拒答测试完成，日志: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
