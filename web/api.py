"""
FastAPI-сервер для веб-интерфейса AI-agent.

Запуск из корня проекта:
    python -m uvicorn web.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импорты из корня работали.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scr.history_store import append_turn, clear_history, load_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic-модели запросов и ответов
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Текст сообщения")
    session_id: str = Field(..., min_length=1, max_length=100, description="ID сессии")

class ChatResponse(BaseModel):
    reply: str
    session_id: str

class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)

# ---------------------------------------------------------------------------
# Приложение FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Agent Web API",
    description="Локальный веб-интерфейс для AI-агента",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Отправить сообщение боту и получить ответ.

    Эндпоинт синхронный (`def`, не `async def`), поэтому FastAPI
    автоматически выполняет его в пуле потоков — это корректно для
    блокирующего вызова LLM/агента.
    """
    logger.info("POST /api/chat  session=%s  len=%d", req.session_id, len(req.message))
    try:
        # Ленивый импорт: чтобы фронтенд открывался даже если LLM не настроен.
        from llm import safe_agent_call_with_history

        history = load_messages(session_id=req.session_id, limit_pairs=10)
        reply = safe_agent_call_with_history(req.message, history)
        append_turn(session_id=req.session_id, user_text=req.message, assistant_text=reply, keep_last_pairs=10)
    except Exception as exc:
        logger.exception("agent failed")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса") from exc
    return ChatResponse(reply=reply, session_id=req.session_id)


@app.get("/api/history")
def get_history(session_id: str, limit_pairs: int = 50):
    """
    Возвращает историю диалога для указанной сессии.
    - session_id: ID сессии (обязательный query-параметр)
    - limit_pairs: количество последних пар сообщений (по умолчанию 50)
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    messages = load_messages(session_id=session_id, limit_pairs=limit_pairs)
    # Преобразуем список HistoryMessage в список словарей
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at
        }
        for msg in messages
    ]


@app.post("/api/clear")
def clear(req: ClearRequest):
    """Очистить историю диалога для данной сессии."""
    logger.info("POST /api/clear  session=%s", req.session_id)
    clear_history(session_id=req.session_id)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Статические файлы (фронтенд) — монтируем ПОСЛЕДНИМ,
# чтобы маршруты /api/* имели приоритет.
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
