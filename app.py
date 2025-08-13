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

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Chatbot API is running"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if request.reset:
            reset_history()
            return {"reply": "History cleared."}
        
        print(f"Processing message: {request.message}")
        reply = get_bot_response(request.message)
        return {"reply": reply}
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {"reply": f"Sorry, I encountered an error: {str(e)}", "error": True}

@app.on_event("shutdown")
def shutdown_event():
    print("Application shutting down")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
