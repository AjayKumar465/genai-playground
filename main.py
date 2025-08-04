from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summarizer import summarize_text

app = FastAPI()

class SummaryRequest(BaseModel):
    text: str
    model: str = "bart"

@app.post("/summarize")
def generate_summary(req: SummaryRequest):
    try:
        summary = summarize_text(req.text, req.model)
        return {"summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
