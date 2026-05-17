# Truth Lens - Fake News Classifier

A FastAPI application that classifies text as fake/misleading or real/credible using HuggingFace transformers and Google Fact Check Tools API.

## Features
- Zero-shot text classification using BART-large-MNLI
- Google Fact Check Tools API integration for additional verification
- FastAPI backend with automatic API documentation

## Model
- **Model Name**: facebook/bart-large-mnli
- **Task**: Zero-shot classification for fake news detection

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Google API Key (Optional)
To enable fact-checking with Google Fact Check Tools API:
1. Get a Google Cloud API key with Fact Check Tools API enabled
2. Set the environment variable:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 4. Run the Application
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/analyze`
Analyzes a text string for fake news detection and provides fact-check results.

**Request:**
```json
{
  "text": "Your text to analyze goes here"
}
```

**Response:**
```json
{
  "label": "REAL",
  "confidence": 0.9854,
  "explanation": "This text has been classified as REAL/CREDIBLE with 98.5% confidence. The language and structure appear consistent with factual reporting. However, always cross-reference with multiple credible sources.",
  "fact_checks": [
    {
      "claim": "Similar claim text",
      "claimant": "Source of the claim",
      "rating": "True/False rating",
      "publisher": "Fact-checking organization",
      "url": "Link to full fact-check"
    }
  ]
}
```
```

### GET `/`
Returns API information and usage example.

## Example Usage

### Using cURL
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "The Earth revolves around the Sun"}'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "Your text to analyze"}
)
print(response.json())
```

### Using FastAPI Interactive Docs
Open `http://localhost:8000/docs` in your browser for Swagger UI documentation.

## Response Format

- **label**: "FAKE" or "REAL"
- **confidence**: Float between 0 and 1 (rounded to 4 decimal places)
- **explanation**: Detailed explanation of the classification with confidence percentage and recommendations

## Notes

- The model runs on CPU by default. Change `device=-1` to `device=0` in `main.py` for GPU acceleration if available.
- First run will download the model (~500MB), subsequent runs will use the cached model.
- The classifier works best with longer text samples (100+ characters).
