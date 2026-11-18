"""建索引 pipeline：chunks JSON → 向量化 → 入库（模型和向量库均可配置）。

Pipeline（与 loaders 的工厂模式配套）：
    阶段 1  加载 chunks    读 CHUNKS_DIR/*.json（切 chunks / chunks_rowwise 做对比）
    阶段 2  转换节点      chunk 字典 → TextNode（metadata 只保留标量，Chroma 限制）
    阶段 3  向量化        get_embedder() ← EMBED_MODEL（bge-m3 本地 / dashscope API）
    阶段 4  入库持久化    get_vector_store() ← VECTOR_STORE（chroma / faiss 预留）
    阶段 5  统计校验      node 数 / 维度 / 索引目录

索引目录隔离：data/index/{模型名}-{向量库名}/{chunk变体名}/
不同组合互不覆盖，检索对比时只切 .env 不改代码。

用法：
    python src/core/build_index.py                          # 默认 bge-m3 + chroma
    EMBED_MODEL=dashscope python src/core/build_index.py    # qwen API 向量化
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llama_index.core.schema import TextNode

from src.core.embedders import get_embedder
from src.core.vectorstores import get_vector_store, vector_store_name

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_INDEX_DIR = ROOT / "data" / "index"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def get_chunks_dir() -> Path:
    """读 CHUNKS_DIR 配置（相对项目根，默认 processed_docs/mineru/chunks）。"""
    return ROOT / os.environ.get("CHUNKS_DIR", "processed_docs/mineru/chunks")


def get_index_dir(embedder_name: str, store_name: str) -> Path:
    """索引目录：不同「模型-向量库-chunk变体」组合各占一个目录，互不覆盖。"""
    chunks_dir = get_chunks_dir()
    return DATA_INDEX_DIR / f"{embedder_name}-{store_name}" / chunks_dir.name


def extract_company_model(filename: str) -> tuple[str, str]:
    """从 chunk 文件名提取（品牌, 车型）。

    文件名格式：{品牌}_{年份}_{车型}_{文档类型}.json
    如 领克_2027_03Plus_配置表.json → ("领克", "03Plus")
        BMW_2024_X3_用户手册.json   → ("BMW", "X3")

    文档集合封闭；解析不出直接报错，提示检查命名，而不是静默漏过滤。
    """
    parts = Path(filename).stem.split("_")
    if len(parts) < 4 or parts[0] not in ("领克", "BMW"):
        raise ValueError(f"无法从文件名解析（品牌, 车型），请检查文件命名: {filename}")
    return parts[0], parts[2]


def load_chunks(chunks_dir: Path) -> list[dict]:
    """阶段 1：加载目录下全部 chunk JSON，并按来源文件附着品牌/车型字段。"""
    files = sorted(chunks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{chunks_dir} 下没有找到 chunk JSON")

    chunks = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            file_chunks = json.load(fp)
        company, model = extract_company_model(f.name)
        for c in file_chunks:
            c["company"] = company
            c["model"] = model
        logger.info("加载 %s: %d 个 chunk（品牌=%s 车型=%s）", f.name, len(file_chunks), company, model)
        chunks.extend(file_chunks)
    return chunks


def to_nodes(chunks: list[dict]) -> list[TextNode]:
    """阶段 2：chunk 字典 → TextNode。

    注意：Chroma 的 metadata 只支持标量（str/int/float/bool），
    所以 path 列表拆成 chapter / section 两个字符串字段；
    company / model 供检索侧路由（Chroma where 前置过滤）。
    """
    return [
        TextNode(
            text=c["text"],
            metadata={
                "company": c["company"],
                "model": c["model"],
                "page": c["page"],
                "chapter": c["chapter"],
                "section": c["section"],
            },
        )
        for c in chunks
    ]


def embed_nodes(embedder, nodes: list[TextNode]) -> list[TextNode]:
    """阶段 3：批量向量化，向量直接填进 node.embedding。

    预填 embedding 后 VectorStoreIndex 建库时不会再向量化，
    保证入库向量全部经过 Embedder 的批量接口（API 模式下省请求数）。
    """
    texts = [n.text for n in nodes]
    logger.info("开始向量化: %d 个文本, 模型=%s", len(texts), embedder.name)
    vectors = embedder.embed_documents(texts)
    for node, vec in zip(nodes, vectors):
        node.embedding = vec
    return nodes


def main():
    start = time.time()

    # 阶段 1：加载 chunks
    chunks_dir = get_chunks_dir()
    logger.info("=" * 60)
    logger.info("阶段 1  加载 chunks: %s", chunks_dir)
    chunks = load_chunks(chunks_dir)

    # 阶段 2：转换节点
    logger.info("阶段 2  转换节点")
    nodes = to_nodes(chunks)

    # 阶段 3：向量化
    embedder = get_embedder()
    logger.info("阶段 3  向量化（EMBED_MODEL=%s）", embedder.name)
    nodes = embed_nodes(embedder, nodes)

    # 阶段 4：入库持久化
    index_dir = get_index_dir(embedder.name, vector_store_name())
    store = get_vector_store(index_dir)
    logger.info("阶段 4  入库（VECTOR_STORE=%s）", store.name)
    store.build(embedder, nodes)

    # 阶段 5：统计校验
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("索引完成")
    logger.info("  node 数:   %d", len(nodes))
    logger.info("  向量维度:  %d", len(nodes[0].embedding))
    logger.info("  索引目录:  %s", index_dir)
    logger.info("  总耗时:    %.1f 秒", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
