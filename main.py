from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)


# === Request Model ===
class AskRequest(BaseModel):
    question: str


# === Prompt Template ===
SYSTEM_PROMPT = """
You are a precise and reliable AI assistant.
Your role is to provide short, accurate, and well-structured answers.

Rules:
- Do NOT hallucinate.
- If you don't know, say "I don't have enough information to answer."
- Keep explanations concise unless asked for details.
- Avoid unnecessary words.
"""


# === API Endpoint ===
@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        response = client.chat.completions.create(
            model="gemma3:4b",
            temperature=0.1,       # Deterministic behavior
            top_p=1,               # No sampling restriction
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.question}
            ]
        )

        answer = response.choices[0].message.content.strip()

        return {
            "success": True,
            "model": "gemma3:4b",
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI model error: {str(e)}"
        )
