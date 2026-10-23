# ============================================================
# BMW X3 车主手册 RAG —— Dockerfile
# 单容器：FastAPI + 静态前端（由 FastAPI StaticFiles 托管）
# 模型缓存通过 volume 持久化，向量库通过宿主机目录挂载
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# 使用国内镜像，避免 deb.debian.org / PyPI 在国内超时或 EOF
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先安装依赖，利用 Docker 层缓存（requirements.txt 不变时不重复安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制应用代码（.dockerignore 会排除 .venv / .env / data / *.pdf 等）
COPY . .

# 为运行时挂载索引准备挂载点（data 本身不进镜像）
RUN mkdir -p /app/data/index

# 统一入口脚本换行符为 LF 并赋予执行权限，避免 Windows 宿主机编辑导致 \r 问题
RUN sed -i 's/\r$//' deploy/entrypoint.sh && chmod +x deploy/entrypoint.sh

# HuggingFace 模型缓存目录（由 docker-compose 挂载到 volume）
ENV HF_HOME=/models \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["deploy/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
