"""带引用和拒答的答案生成"""
import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.deepseek import DeepSeek
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

    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.3,
        messages=[{"role":"user", "content":prompt}],
    )
    return resp.choices[0].message.content


def generate_stream(question :str, contexts):
    """流式输出：逐块 yield 出答案文本 
    和 generate（）的区别：多了 stream=True, 并把结果逐块 yield
    """
    from openai import OpenAI

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

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    stream = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.3,
        messages=[{"role":"user", "content":prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta