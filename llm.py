import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_llm(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY not set"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",  # 🔥 strong + impressive
        "messages": [
            {"role": "system", "content": "Answer strictly from the given context. If unsure, say you don't know."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    resp = requests.post(GROQ_URL, headers=headers, json=payload)

    if resp.status_code != 200:
        return f"LLM Error {resp.status_code}: {resp.text}"

    return resp.json()["choices"][0]["message"]["content"]
