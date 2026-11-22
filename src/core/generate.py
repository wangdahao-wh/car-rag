"""带引用和拒答的答案生成；对比问题改写成单车型检索。"""
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _deepseek_client() -> OpenAI:
    """DeepSeek OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


def rewrite_compare_query(query: str, models: list[str], companies: list[str]) -> list[str]:
    """把对比问题改写成多个单车型/品牌问题，供分别检索。

    只改写，不做检索。每个输出问题只保留一个车型或品牌。
    """
    prompt = f"""把对比问题改写成多个只含一个车型的独立问题。
规则：
1. 每个问题只出现一个车型，保持原问题的信息需求；
2. 不要新增原问题没有的信息需求；
3. **关键**：如果原问题问的是"有什么"、"有哪些"、"包括什么"等枚举类信息，改写后必须保留"有哪些"等枚举词，不能改成"是什么"；
4. 已知车型：{models}；已知品牌：{companies}；
5. 只输出 JSON 字符串数组，不要解释。

问题：{query}

以下是举例：
问题：领克 03+ 和 宝马 m240 的最高时速分别是多少？
改写：["领克 03+ 的最高时速是多少", "宝马 m240 的最高时速是多少"]

问题：2系和X3的安全配置有什么区别？
改写：["2系有哪些安全配置", "X3有哪些安全配置"]"""
    resp = _deepseek_client().chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError(f"对比改写未返回 JSON 数组: {text}")
    rewritten = json.loads(match.group())
    if not isinstance(rewritten, list) or not all(isinstance(q, str) and q.strip() for q in rewritten):
        raise ValueError(f"对比改写结果非法: {rewritten}")
    return rewritten


def generate(question, contexts):
    """基于检索到的上下文生成答案
    关键：prompt里强制约束「只依据资料 + 没有就说无法确定」"""

    ctx = "\n\n".join(
        f"[{i}](第{n.node.metadata['page']}页) {n.node.text}" for i, n in enumerate(contexts, 1)
    )

    prompt = f"""你是严谨的 BMW 车主手册助手。请严格遵守：
1) 只能依据【参考资料】回答，资料里没有的信息，直接说「根据现有资料无法确定」，绝不编造；
2) 用到资料的地方，用 [编号] 标注来源；
3) 回答简洁，用词通俗。

【参考资料】
{ctx}

【问题】
{question}"""

    resp = _deepseek_client().chat.completions.create(
        model="deepseek-chat",
        temperature=0.3,
        messages=[{"role":"user", "content":prompt}],
    )
    return resp.choices[0].message.content


def generate_stream(question :str, contexts):
    """流式输出：逐块 yield 出答案文本 
    和 generate（）的区别：多了 stream=True, 并把结果逐块 yield
    """
    ctx = "\n\n".join(
        f"[{i}](第 {n.node.metadata['page']}页) {n.node.text}"
        for i, n in enumerate(contexts, 1)
    )

    prompt = f"""你是严谨的 BMW 车主手册助手。请严格遵守：
    1) 只能依据【参考资料】回答，资料里没有的信息，直接说「根据现有资料无法确定」，绝不编造；
    2) 用到资料的地方，用 [编号] 标注来源；
    3) 回答简洁，用词通俗。

    【参考资料】
    {ctx}

    【问题】
    {question}"""

    stream = _deepseek_client().chat.completions.create(
        model="deepseek-chat",
        temperature=0.3,
        messages=[{"role":"user", "content":prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta