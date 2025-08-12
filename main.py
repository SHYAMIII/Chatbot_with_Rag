# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import get_bot_response, reset_history
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS so Next.js can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Next.js domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    reset: bool = False  # default False so old clients still work


@app.post("/chat")
async def chat(request: ChatRequest):
    if request.reset:   # if frontend asks to clear
        reset_history()
        return {"reply": "History cleared."}  # stop here, don't call get_bot_response
    
    reply = get_bot_response(request.message)
    return {"reply": reply}


if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
