from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_ai(payload: AskRequest):
    response = client.chat.completions.create(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload.question},
        ]
    )
    return {"answer": response.choices[0].message.content}
