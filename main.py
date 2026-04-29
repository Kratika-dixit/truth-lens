from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="TruthLens - Fake News Classifier", version="1.0")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

class TextInput(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    label: str
    confidence: float
    explanation: str

def get_explanation(label: str, confidence: float) -> str:
    confidence_pct = confidence * 100
    if label == "FAKE":
        return (
            f"This text has been classified as FAKE/MISLEADING "
            f"with {confidence_pct:.1f}% confidence. "
            "The language patterns suggest potential misinformation. "
            "Verify through credible fact-checking sources."
        )
    else:
        return (
            f"This text has been classified as REAL/CREDIBLE "
            f"with {confidence_pct:.1f}% confidence. "
            "The language appears consistent with factual reporting. "
            "Always cross-reference with multiple credible sources."
        )

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_text(input_data: TextInput) -> AnalysisResponse:
    if not input_data.text or len(input_data.text.strip()) == 0:
        raise ValueError("Text input cannot be empty")

    result = classifier(
        input_data.text,
        candidate_labels=["fake news", "real news"]
    )

    model_label = "FAKE" if result["labels"][0] == "fake news" else "REAL"
    confidence_score = result["scores"][0]
    explanation = get_explanation(model_label, confidence_score)

    return AnalysisResponse(
        label=model_label,
        confidence=round(confidence_score, 4),
        explanation=explanation
    )

@app.get("/")
def root():
    return {
        "name": "TruthLens",
        "description": "Fake News Classifier using HuggingFace",
        "endpoint": "/analyze",
        "method": "POST",
        "example": {
            "text": "Breaking: Major new discovery announced!"
        }
    }