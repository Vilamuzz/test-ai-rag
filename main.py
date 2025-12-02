from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure Ollama-compatible client
client = AsyncOpenAI(
    base_url=os.getenv("BASE_URL"), 
    api_key=os.getenv("API_KEY")
)

class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: AskRequest):
    try:
        response = await client.chat.completions.create(
            model="gemma3:4b",
            messages=[
                {"role": "system", "content": "You are a concise, helpful assistant."},
                {"role": "user", "content": request.question}
            ]
        )

        answer = response.choices[0].message.content
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
