"""MinerU 在线 API 交互 + qwen-vl 表格转录（全部网络 IO）。

从 mineru_loader.py 拆出：PDF 分片、批量上传、轮询下载、视觉模型读图转录。
split_pdf 依赖 pypdf 的 PdfReader/PdfWriter（本文件内 import）。
"""
import base64
import logging
import re
import time
import zipfile
from pathlib import Path

import requests
from pypdf import PdfReader, PdfWriter

MAX_PAGES = 200          # MinerU 单文件页数上限
POLL_INTERVAL = 10       # 轮询间隔（秒）
VL_MODEL = "qwen3-vl-plus"  # 空表体二次提取用的视觉模型（qwen-vl-ocr 会丢弃符号单元格，不能用）

# 空表体转录提示词：图片型/符号型配置表转 HTML
TABLE_TRANSCRIBE_PROMPT = (
    "把图片中的表格完整转录为 HTML <table>，只输出表格本身，不要输出任何解释。"
    "最重要的要求：表格中的符号单元格（实心圆点、空心圆点、短横线）必须逐格转录，"
    "绝对不能省略——每个配置项所在行，在每个车型列下都必须有对应的单元格，"
    "内容是●（标配）、O（选装）或-（不含）其中一个。"
    "其他要求：1. 单元格内容原样转录；2. 合并单元格用 rowspan/colspan 表示；"
    "3. 表头和所有行列都要完整，不要遗漏。"
)

logger = logging.getLogger(__name__)


def split_pdf(pdf_path: Path, output_dir: Path, max_pages: int = MAX_PAGES) -> list[tuple[Path, int, int]]:
    """把超过页数上限的 PDF 切成多个分片，写入 output_dir（不污染源文档目录）。

    Returns:
        [(分片路径, 起始页, 结束页), ...]；未超限时返回 [(原路径, 1, 总页数)]（页码 1 起）
    """
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    if total <= max_pages:
        return [(pdf_path, 1, total)]

    output_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, total, max_pages):
        writer = PdfWriter()
        for page in reader.pages[start:start + max_pages]:
            writer.add_page(page)
        end = min(start + max_pages, total)
        part_path = output_dir / f"{pdf_path.stem}_part{start // max_pages + 1}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)
        parts.append((part_path, start + 1, end))
        logger.info("已切分: %s（第 %d-%d 页）", part_path.name, start + 1, end)
    return parts


def create_batch(api_key: str, file_paths: list[Path]) -> tuple[str, list[str]]:
    """申请批量上传，返回 (batch_id, 预签名上传地址列表)。"""
    res = requests.post(
        "https://mineru.net/api/v4/file-urls/batch",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "enable_formula": False,
            "language": "ch",
            "files": [{"name": p.name} for p in file_paths],
        },
        timeout=30,
    )
    payload = res.json()
    return payload["data"]["batch_id"], payload["data"]["file_urls"]


def upload_files(file_paths: list[Path], upload_urls: list[str]) -> None:
    """把本地 PDF 逐个 PUT 到预签名地址（地址自带权限，无需鉴权头）。"""
    for path, url in zip(file_paths, upload_urls):
        with open(path, "rb") as f:
            res = requests.put(url, data=f, timeout=300)
        logger.info("上传 %s: %d", path.name, res.status_code)


def download_results(api_key: str, batch_id: str, output_dir: Path) -> dict[str, Path]:
    """轮询批量任务直到全部进入终态，下载 zip 并解压。

    Returns:
        {文件名: 解压目录}；失败的任务不在结果里

    注：zip 压缩包保留在 output_dir，解压目录为 output_dir/{文件名}/
    """
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # 轮询直到所有文件 done 或 failed
    while True:
        results = requests.get(url, headers=headers, timeout=30).json()["data"]["extract_result"]
        for item in results:
            logger.info("%s: %s", item.get("file_name"), item.get("state"))
        if all(item.get("state") in ("done", "failed") for item in results):
            break
        logger.info("任务未完成，%d 秒后重试...", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)

    output_dir.mkdir(parents=True, exist_ok=True)
    extract_dirs = {}
    for item in results:
        if item.get("state") != "done":
            continue
        name = item["file_name"]
        zip_path = output_dir / f"{name}.zip"

        # 下载 zip 压缩包
        r = requests.get(item["full_zip_url"], stream=True, timeout=300)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("已下载: %s", zip_path)

        # 解压到同名目录
        extract_dir = output_dir / Path(name).stem
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("已解压: %s", extract_dir)

        extract_dirs[name] = extract_dir
    return extract_dirs


def transcribe_table_image(image_path: Path, api_key: str, vl_url: str) -> str:
    """调用 qwen-vl 把表格图片转录为 HTML 表格（表体不可用二次提取）。

    MinerU 对图片型/符号型表格的两种失败形态（空表体/网格挤压），
    表格内容只在裁剪图片里，需要视觉模型重新读图转录。
    """
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    res = requests.post(
        vl_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": VL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": TABLE_TRANSCRIBE_PROMPT},
                    ],
                }
            ],
        },
        timeout=300,
    )
    res.raise_for_status()
    html = res.json()["choices"][0]["message"]["content"].strip()
    # 去掉模型可能包裹的 markdown 代码块
    html = re.sub(r"^```(?:html)?\s*|\s*```$", "", html)
    if "<table" not in html:
        raise RuntimeError(f"qwen-vl 转录结果不是表格: {html[:200]}")
    return html
