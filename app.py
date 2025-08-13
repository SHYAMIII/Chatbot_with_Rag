# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import get_bot_response, reset_history
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI()

# Allow CORS so frontend can access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    reset: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.reset:
        reset_history()
        return {"reply": "History cleared."}
    reply = get_bot_response(request.message)
    return {"reply": reply}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
