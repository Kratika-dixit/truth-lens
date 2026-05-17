from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import requests
import os

app = FastAPI(title="TruthLens - Fake News Classifier", version="1.0")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

class TextInput(BaseModel):
    text: str

class FactCheckResult(BaseModel):
    claim: str
    claimant: str
    rating: str
    publisher: str
    url: str

class AnalysisResponse(BaseModel):
    label: str
    confidence: float
    explanation: str
    fact_checks: list[FactCheckResult]

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

def check_facts_with_google(text: str) -> list[FactCheckResult]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return []
    
    # Extract a query from the text (first sentence or first 100 characters)
    query = text.split('.')[0][:100] if '.' in text else text[:100]
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": query,
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        fact_checks = []
        if "claims" in data:
            for claim in data["claims"][:3]:  # Limit to top 3 results
                claim_review = claim.get("claimReview", [{}])[0]
                fact_checks.append(FactCheckResult(
                    claim=claim.get("text", ""),
                    claimant=claim.get("claimant", ""),
                    rating=claim_review.get("textualRating", ""),
                    publisher=claim_review.get("publisher", {}).get("name", ""),
                    url=claim_review.get("url", "")
                ))
        return fact_checks
    except Exception as e:
        print(f"Error querying Google Fact Check API: {e}")
        return []

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
    fact_checks = check_facts_with_google(input_data.text)

    return AnalysisResponse(
        label=model_label,
        confidence=round(confidence_score, 4),
        explanation=explanation,
        fact_checks=fact_checks
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