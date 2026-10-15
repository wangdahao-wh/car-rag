CHAPTERS = {
  "便捷查阅",
  "提示",
  "操作",
  "简要说明",
  "顺利驾驶",
  "驾驶提示"}

def clean_page(raw: str, page_no: int):
    lines = raw.split("\n")

    while lines and not lines[-1].strip():
        lines.pop()

    start = None
    for i in range(len(lines)-1,-1,-1):
        if lines[i].startswith("Seite "):
            start = i
            break

    # 如果有“Seite”，则将其作为页脚，去掉页脚内容
    if start is not None:
        body = lines[:start]
        raw_parts = [l.strip() for l in lines[start+1:-2] if l.strip()]
        breadcrumb = normalize_breadcrumb(raw_parts)
    elif lines and lines[-1].startswith("Online Edition"):
        # 兜底：至少删掉版本号行
        body = lines[:-1]
        if body and body[-1].strip().isdigit():
            body = body[:-1]
        breadcrumb = []
    else:
        # 连版本号都没有，整页当正文
        body = lines
        breadcrumb = []


    text = "\n".join(body).strip()
    meta = {"page": page_no, "breadcrumb":breadcrumb}
    return text, meta

def normalize_breadcrumb(parts):
    if len(parts) == 2:
        return parts
    if len(parts) == 1:
        text = parts[0]
        for c in CHAPTERS:
            if text.endswith(c):
                section = text[:-len(c)].strip()
                return [c, section]
    return parts
