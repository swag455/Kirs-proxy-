from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
import re
import os
from typing import List, Dict, Any

app = FastAPI(title="Kirs Ultimate Proxy")

# API ключи берутся из переменных окружения (настроим позже в Render)
GROQ_KEY = os.environ.get("GROQ_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_KEY", "")
MISTRAL_KEY = os.environ.get("MISTRAL_KEY", "")

# 7 лучших моделей с приоритетами
MODELS = [
    # FAST - быстрые ответы
    {"id": "llama-3.1-8b-instant", "provider": "groq", 
     "api_url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY},
    {"id": "gemini-2.0-flash-lite", "provider": "google", 
     "api_url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_KEY}", "key": GEMINI_KEY},
    # SMART - сложные рассуждения
    {"id": "gemini-3.5-flash", "provider": "google", 
     "api_url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_KEY}", "key": GEMINI_KEY},
    {"id": "llama-3.3-70b-versatile", "provider": "groq", 
     "api_url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY},
    # VISION - фото
    {"id": "llama-4-scout-17b-16e-instruct", "provider": "groq", 
     "api_url": "https://api.groq.com/openai/v1/chat/completions", "key": GROQ_KEY},
    # DEEP - огромные тексты
    {"id": "mistral-large-latest", "provider": "mistral", 
     "api_url": "https://api.mistral.ai/v1/chat/completions", "key": MISTRAL_KEY},
]

def is_rate_limit(error_text: str) -> bool:
    """Проверяет, является ли ошибка превышением лимита"""
    return any(x in error_text.lower() for x in ["rate limit", "429", "quota", "rpd", "too many"])

async def call_model(model: dict, messages: List[Dict]) -> tuple:
    """Отправляет запрос к модели. Возвращает (response_json, status_code, model_id)"""
    headers = {"Authorization": f"Bearer {model['key']}", "Content-Type": "application/json"}
    
    if model["provider"] == "google":
        # Формат Google Gemini
        contents = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages]
        payload = {"contents": contents}
    else:
        # Формат OpenAI (Groq, Mistral)
        payload = {"model": model["id"], "messages": messages, "temperature": 0.7}
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(model["api_url"], json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json(), resp.status_code, model["id"]
        else:
            return {"error": resp.text}, resp.status_code, model["id"]

@app.post("/v1/chat/completions")
async def chat(request: Request):
    """Главный эндпоинт, который принимает запросы от Open WebUI"""
    body = await request.json()
    messages = body.get("messages", [])
    
    # Пробуем модели по очереди
    for model in MODELS:
        result, status, model_id = await call_model(model, messages)
        
        if status == 200:
            # Успех — возвращаем ответ с меткой модели
            if "candidates" in result:
                # Google Gemini формат
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return {"choices": [{"message": {"content": f"[Модель: {model_id}]\n\n{text}"}}]}
            else:
                # OpenAI формат (Groq, Mistral)
                content = result["choices"][0]["message"]["content"]
                result["choices"][0]["message"]["content"] = f"[Модель: {model_id}]\n\n{content}"
                return result
        
        # Если ошибка лимита — пробуем следующую модель
        if is_rate_limit(str(result)):
            print(f"[Proxy] Лимит {model_id}, пробую следующую...")
            continue
        
        # Другая ошибка — возвращаем как есть
        return JSONResponse(status_code=status, content=result)
    
    # Все модели исчерпали лимиты
    return {"error": "Все модели превысили лимиты. Попробуйте позже."}

@app.get("/health")
async def health():
    """Проверка работоспособности"""
    return {"status": "ok", "models_count": len(MODELS)}

@app.get("/")
async def root():
    return {"message": "Kirs Ultimate Proxy is running", "status": "active"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)