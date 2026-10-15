"""FastAPI 接口层"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.rag_service import RAGService

ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = ROOT / "data" / "index"

# 全局服务实例 启动时初始化，请求时复用
service : RAGService | None = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    """启动时加载模型， 关闭时清理"""
    global service
    print("⏳ 正在加载模型（bge-m3 + reranker），首次启动需要几十秒...")
    service = RAGService(INDEX_DIR)
    print("✅ 服务就绪")
    yield
    print("👋 服务关闭")

app = FastAPI(
    title="BMW 车主手册问答",
    description="基于 LlamaIndex + bge-m3 + ChromaDB 的 RAG 问答系统",
    version="1.0",
    lifespan=lifespan,
)

# -------------- 数据模型 (Pydantic 自动校验) --------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=200, description="用户问题")

class Source(BaseModel):
    page : int
    chapter: str
    section : str
    text : str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

# ------ 接口 ------

@app.get("/health")
def health():
    """健康检查"""
    return {"status" : "ok", "ready" : service is not None}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """问答接口"""
    if service is None:
        raise HTTPException(status_code=503, detail="服务正在启动，请稍后再试")
    return service.ask(req.question)

from fastapi.responses import StreamingResponse

@app.post("/ask/stream")
def ask_stream(req:AskRequest):
    """流式问答 (SSE) """
    if service is None:
        raise HTTPException(status_code=503, detail="服务正在启动")

    return StreamingResponse(
        service.ask_stream(req.question),
        media_type="text/event-stream",
    )
app.mount("/", StaticFiles(directory=str(ROOT/ "static"), html=True), name = "static")

