
def extract_blocks(page):
    """用字体信息抽取结构化片段，并剔除页脚"""
    parts = []
    def visitor(text, cm, tm, font_dict, font_size):
        t = text.strip()
        if t:
            parts.append((round(font_size, 1), t))
    page.extract_text(visitor_text=visitor)

    # 剔除页脚：从后往前找“Seite”片段，那里之后的全是页脚
    for i in range(len(parts)-1,-1,-1):
        if parts[i][1].startswith("Seite "):
            parts = parts[:i]
            break

    return parts

# def build_sections(parts):
#     """按标题（字号>=10)切分成小节"""
#     sections = []
#     title, lines = None, []

#     for size, text in parts:
#         if size >= 10.0:
#             if lines:
#                 sections.append((title, lines))
#             title, lines = text, []
#         else:
#             lines.append(text)

#     if lines:
#         sections.append((title, lines))
#     return sections
SPLIT_SIZE = 11.0
HEADER_SIZE = 10.0

def build_sections(parts : list):
    """按标题（字号>=10)切分成小节"""
    sections = []
    titles, lines = [], [] # title为标题栈，记录达到当前位置的完成路径

    for size, text in parts:
        if size >= HEADER_SIZE:
            if size >=  SPLIT_SIZE and lines:
                sections.append((list(titles), lines))
                lines = []
            titles = [t for t in titles if t[0] > size] + [(size, text)]
        else:
            lines.append(text)

    if lines:
        sections.append((list(titles), lines))
    return sections

def section_text(titles, lines):
    """把标题路径 + 正文 拼成 chunk 文本"""
    path = ' > '.join(t for _, t in titles)
    body = join_lines(lines)
    return f"[{path}]\n[{body}]" if path else body


def join_lines(lines):
    """把 PDF 执行的片段拼回完整句子"""
    out = ""
    for l in lines:
        if out and out[-1] not in "。！？：；": # 上一段没说完，直接接上
            out += l
        else:
            out += ("\n" if out else "") + l
    return out


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
PDF_PATH = Path(__file__).resolve().parent.parent.parent / "docs/x3-2024.pdf"

from statistics import median

if __name__ == "__main__":
    reader = PdfReader(PDF_PATH)
    for pno in [155,115]:
        parts = extract_blocks(reader.pages[pno-1])
        print(f"=== 第{pno}页的标题===")
        for size, text in parts:
            if size >= 10.0:
                print(f" [{size} {text}]")
        sections = build_sections(parts)
        lens = []
        print(f"sections 共 {len(sections)} 个")
        for i, (titles, lines) in enumerate(sections, 1):
            text = section_text(titles, lines)
            lens.append(len(text))
            print(f"\n--小节 {i} ({len(text)} 字) ---")
            print(text)
        print(f"\n字数: 最小 {min(lens)} / 中位 {int(median(lens))} / 最大 {max(lens)}")

            # for thresh in [10.0, 11.0, 14.0]:
            #     sections = build_sections(parts, thresh)
            #     lens = [len(join_lines(l)) for _,l in sections]
            #     print(f"\n阈值 {thresh}: 切出 {len(sections)} 节"
            #           f"中位数 {int(median(lens))} 字，最大 {max(lens)} 字")
            #     # print(f"\n===== 切出 {len(sections)} 个小节 =====")
            #     for i, (title, lines) in enumerate(sections,1):
            #         text = join_lines(lines)
            #         print(f"\n --小节{i} [标题： {title}] ({len(text)}个字)")
            #         print(text)

            # # ③ 看字数分布
            # lens = [len(join_lines(lines)) for _, lines in sections]
            # print(f"\n===== 小节字数分布 =====")
            # print(f"  最小 {min(lens)} / 中位 {int(median(lens))} / 最大 {max(lens)}")
            # print(f"  < 100 字的小节: {sum(1 for x in lens if x < 100)} 个")
            # print(f"  > 800 字的小节: {sum(1 for x in lens if x > 800)} 个")

