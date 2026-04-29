# Truth Lens - Fake News Classifier

A FastAPI application that classifies text as fake/misleading or real/credible using HuggingFace transformers.

## Model
- **Model Name**: hamzab/roberta-fake-news-classifier
- **Base**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task**: Text Classification for Fake News Detection

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

### 3. Run the Application
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/analyze`
Analyzes a text string for fake news detection.

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
  "explanation": "This text has been classified as REAL/CREDIBLE with 98.5% confidence. The language and structure appear consistent with factual reporting. However, always cross-reference with multiple credible sources."
}
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
