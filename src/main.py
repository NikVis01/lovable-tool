from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os
import httpx

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
CHAT_CSV = DATA_DIR / "chat_output.csv"

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://langflow:7860")
FLOW_ID = os.getenv("LANGFLOW_FLOW_ID", "")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    response: str

@app.get("/api/candidates")
def get_candidates():
    if not CANDIDATES_CSV.exists():
        raise HTTPException(status_code=404, detail="candidates.csv not found")
    df = pd.read_csv(CANDIDATES_CSV)
    return {"rows": df.to_dict(orient="records")}

@app.get("/api/chat/history")
def get_chat_history(limit: int = 100):
    if not CHAT_CSV.exists():
        return {"rows": []}
    df = pd.read_csv(CHAT_CSV)
    if limit:
        df = df.tail(limit)
    return {"rows": df.to_dict(orient="records")}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not FLOW_ID:
        raise HTTPException(status_code=500, detail="LANGFLOW_FLOW_ID not set")
    endpoint = f"{LANGFLOW_URL}/api/v1/run/{FLOW_ID}?stream=false"
    payload = {
        "input_value": req.message,
        "output_type": "chat",
        "input_type": "chat",
        "tweaks": {},
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(endpoint, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Langflow error: {e}")

    # Best-effort parse: try common locations for the generated text
    resp_text = (
        data.get("outputs", [{}])[0]
            .get("outputs", [{}])[0]
            .get("results", {})
            .get("message", {})
            .get("text", "")
        or data.get("output", "")
        or str(data)
    )

    # Append to chat csv
    try:
        row = {"message": req.message, "response": resp_text}
        if CHAT_CSV.exists():
            df = pd.read_csv(CHAT_CSV)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(CHAT_CSV, index=False)
    except Exception:
        pass

    return ChatResponse(message=req.message, response=resp_text)

# Optional: health
@app.get("/healthz")
def health():
    return {"ok": True}