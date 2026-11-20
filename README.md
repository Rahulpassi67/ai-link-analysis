
# AI-Powered Website Link Analysis API

A Python-based REST API that analyzes website URLs and generates AI-powered business insights using Google Gemini and FastAPI.

## Features

- Website URL validation and content extraction
- AI-powered competitor analysis
- Target search query recommendations
- SEO keyword and metadata recommendations
- Top five service identification
- Target audience analysis
- Structured JSON responses
- Error handling and automated tests
- Automatic retries for temporary Gemini API errors

## Tech Stack

- Python 3
- FastAPI
- Google Gemini API
- HTTPX
- BeautifulSoup
- Pydantic
- Pytest

## Project Structure

```text
ai-link-analysis/
├── app/
│   ├── services/
│   │   ├── llm_service.py
│   │   └── website_extractor.py
│   ├── utils/
│   ├── main.py
│   └── config.py
├── tests/
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ai-link-analysis
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Never commit your actual API key.

## Run the API

```bash
python3 -m uvicorn app.main:app --reload --port 4000
```

API base URL:

http://127.0.0.1:4000

Interactive Swagger documentation:

http://127.0.0.1:4000/docs

## API Usage

### POST /link-analysis

Request:

```json
{
  "url": "https://www.hubspot.com/"
}
```

The API returns structured JSON containing:

1. Competitor analysis
2. Target search queries
3. SEO analysis
4. Top services
5. Target audience

## Run Automated Tests

```bash
python3 -m pytest -v
```

## Environment Variables

| Variable | Description |
|---|---|
| GEMINI_API_KEY | Google Gemini API key |
| GEMINI_MODEL | Gemini model identifier |

## Notes

- Analysis quality depends on the content accessible on the submitted website.
- Recommendations and inferred information should be distinguished from verified website facts.
- Search volumes and numerical performance claims are not fabricated.
- API access depends on the availability and limits of the selected Gemini model.

## Author

Rahul Passi

GitHub: https://github.com/Rahulpassi67
