from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from typing import List, Dict, Any

app = FastAPI(title="Kirs Ultimate Proxy")

# ========== РАЗРЕШАЕМ CORS (ВАЖНО ДЛЯ HF SPACE) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== API КЛЮЧИ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ ==========
GROQ_KEY = os.environ.get("GROQ_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
MISTRAL_KEY = os.environ.get("MISTRAL_KEY", "")

# ========== 7 МОДЕЛЕЙ С ПРИОРИТЕТАМИ ==========
MODELS = [
    # FAST
    {
        "id": "llama-3.1-8b-instant",
        "provider": "groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "key": GROQ_KEY
    },
    {
        "id": "gemini-2.0-flash-lite",
        "provider": "google",
        "api_url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_KEY}",
        "key": GEMINI_KEY
    },
    # SMART
    {
        "id": "gemini-3.5-flash",
        "provider": "google",
        "api_url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_KEY}",
        "key": GEMINI_KEY
    },
    {
        "id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "key": GROQ_KEY
    },
    # VISION
    {
        "id": "llama-4-scout-17b-16e-instruct",
        "provider": "groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "key": GROQ_KEY
    },
    # DEEP
    {
        "id": "mistral-large-latest",
        "provider": "mistral",
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "key": MISTRAL_KEY
    },
]

MODEL_IDS = [m["id"] for m in MODELS]

# ========== ФУНКЦИЯ ПРОВЕРКИ ОШИБКИ ЛИМИТА ==========
def is_rate_limit(error_text: str) -> bool:
    keywords = ["rate limit", "429", "quota", "rpd", "too many", "exceeded", "rpm"]
    return any(kw in error_text.lower() for kw in keywords)

# ========== ФУНКЦИЯ ВЫЗОВА МОДЕЛИ ==========
async def call_model(model: dict, messages: List[Dict]) -> tuple:
    headers = {"Authorization": f"Bearer {model['key']}", "Content-Type": "application/json"}
    
    if model["provider"] == "google":
        # Gemini формат
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            content = m.get("content", "")
            contents.append({"role": role, "parts": [{"text": content}]})
        payload = {"contents": contents}
    else:
        # OpenAI формат (Groq, Mistral)
        payload = {
            "model": model["id"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096
        }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(model["api_url"], json=payload, headers=headers)
            if resp.status_code == 200:
                return resp.json(), resp.status_code, model["id"]
            else:
                return {"error": resp.text}, resp.status_code, model["id"]
        except Exception as e:
            return {"error": str(e)}, 500, model["id"]

# ========== ЭНДПОИНТЫ ==========

@app.get("/health")
async def health():
    return {"status": "ok", "models_count": len(MODELS), "models": MODEL_IDS}

@app.get("/v1/models")
async def list_models():
    """Список моделей для Open WebUI"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "kirs-proxy"
            }
            for model_id in MODEL_IDS
        ]
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    """Основной эндпоинт для запросов"""
    body = await request.json()
    messages = body.get("messages", [])
    
    if not messages:
        return JSONResponse(status_code=400, content={"error": "No messages"})
    
    # Пробуем модели по очереди
    for model in MODELS:
        result, status, model_id = await call_model(model, messages)
        
        if status == 200:
            # Успешный ответ
            if "candidates" in result:
                # Gemini формат
                try:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    text = "Ошибка парсинга ответа Gemini"
                return {
                    "choices": [{
                        "message": {"content": f"[Модель: {model_id}]\n\n{text}"}
                    }]
                }
            else:
                # OpenAI формат
                try:
                    content = result["choices"][0]["message"]["content"]
                    result["choices"][0]["message"]["content"] = f"[Модель: {model_id}]\n\n{content}"
                except (KeyError, IndexError):
                    pass
                return result
        
        # Если ошибка лимита — пробуем следующую модель
        if is_rate_limit(str(result)):
            print(f"[Proxy] Лимит на {model_id}, пробую следующую...")
            continue
        
        # Другая ошибка — возвращаем как есть
        return JSONResponse(status_code=status, content=result)
    
    # Все модели исчерпали лимиты
    return {
        "choices": [{
            "message": {
                "content": "⚠️ Все модели временно недоступны (превышены лимиты). Попробуйте через 5-10 минут.\n\nДоступные модели:\n" + "\n".join([f"- {m['id']}" for m in MODELS])
            }
        }]
    }

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
