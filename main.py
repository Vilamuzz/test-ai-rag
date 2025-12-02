from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

from rag.vectorstore import search
from rag.documents import documents

load_dotenv()

app = FastAPI()

client = AsyncOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

class AskRagRequest(BaseModel):
    question: str


@app.post("/ask-rag")
async def ask_rag(request: AskRagRequest):
    try:
        # Step 1: Retrieve relevant docs
        doc_ids = search(request.question, k=3)
        retrieved = "\n\n".join([documents[i] for i in doc_ids])

        # Step 2: Build augmented prompt
        system_prompt = (
            "You are an AI assistant that answers questions using only the retrieved context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{retrieved}\n\n"
        )

        # Step 3: Ask Gemma model with context
        response = await client.chat.completions.create(
            model="gemma3:4b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ]
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "context_used": retrieved
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
