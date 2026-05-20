from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests

app = FastAPI(title="TruthLens", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

GOOGLE_API_KEY = "AIzaSyCzMRa3rtZykQH27W9sN04Lj1fG-3CK3nQ"

class TextInput(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    found: bool
    results: list
    message: str

def google_fact_check(text: str):
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": text,
        "key": GOOGLE_API_KEY,
        "languageCode": "en"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        claims = data.get("claims", [])
        if not claims:
            return []
        results = []
        for claim in claims[:5]:
            review = claim.get("claimReview", [{}])[0]
            results.append({
                "claim": claim.get("text", ""),
                "rating": review.get("textualRating", ""),
                "source": review.get("publisher", {}).get("name", ""),
                "url": review.get("url", "")
            })
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_text(input_data: TextInput):
    if not input_data.text or len(input_data.text.strip()) == 0:
        raise ValueError("Text input cannot be empty")

    results = google_fact_check(input_data.text)

    if results:
        return AnalysisResponse(
            found=True,
            results=results,
            message="Fact checks found from credible sources."
        )
    else:
        return AnalysisResponse(
            found=False,
            results=[],
            message="No fact checks found. Try a specific news headline."
        )

@app.get("/")
def read_root():
    return FileResponse("static/index.html")