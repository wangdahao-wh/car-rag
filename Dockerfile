# ============================================================
# BMW X3 车主手册 RAG —— Dockerfile
# 单容器：FastAPI + 静态前端（由 FastAPI StaticFiles 托管）
# 模型缓存与向量库通过 volume 持久化
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖：build-essential 用于编译部分 Python 包；curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先安装依赖，利用 Docker 层缓存（requirements.txt 不变时不重复安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码（.dockerignore 会排除 .venv / .env / data/index / *.pdf 等）
COPY . .

# 统一入口脚本换行符为 LF 并赋予执行权限，避免 Windows 宿主机编辑导致 \r 问题
RUN sed -i 's/\r$//' deploy/entrypoint.sh && chmod +x deploy/entrypoint.sh

# HuggingFace 模型缓存目录（由 docker-compose 挂载到 volume）
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["deploy/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
