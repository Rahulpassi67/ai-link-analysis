# Linkly AI

Linkly AI analyzes a public website and returns structured business, SEO, product/service, audience, and competitor insights. It uses FastAPI, Gemini, Tavily, HTTPX, BeautifulSoup, and scikit-learn TF-IDF similarity. The frontend is served by the same FastAPI app.

## Local setup

Use Python 3.12 (also specified in `.python-version`). From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` and set your Gemini key; add a Tavily key to enable competitor discovery. Keep `.env` private; it is ignored by Git. Gemini is required for the core analysis. Tavily discovery is optional, so the API returns the website analysis with an unavailable discovery status if Tavily is not configured or fails. The health endpoint and frontend remain available without provider keys.

Start the application:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 4000
```

Open <http://127.0.0.1:4000/> for the dashboard, <http://127.0.0.1:4000/health> for the health check, and <http://127.0.0.1:4000/docs> for the API documentation.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes, for analysis | Authenticates Gemini website analysis and competitor extraction. |
| `TAVILY_API_KEY` | No, optional competitor discovery | Authenticates competitor web search. |
| `GEMINI_MODEL` | No | Gemini model identifier; defaults to `gemini-3.8-flash`. |

Missing provider keys raise a clear configuration error when analysis is requested (HTTP 503). They do not prevent app imports, the dashboard, or `/health` from working.

## API

### `POST /link-analysis`

Request:

```json
{
  "url": "https://www.example.com/"
}
```

The response includes extracted website metadata and generated analysis. Competitor discovery adds search-derived candidates, heuristic semantic ranking, and webpage evidence checks. Similarity and evidence scores are heuristic signals, not calibrated probabilities or proof of a competitor relationship.

## Tests

Run the suite from the project root:

```bash
python -m pytest -v
```

## Deploy to Render

The repository includes a Render Blueprint in `render.yaml`. Create a Blueprint from the repository in Render and provide `GEMINI_API_KEY` when prompted. `TAVILY_API_KEY` is optional; add it in the Render service's Environment settings only if competitor search is enabled. `GEMINI_MODEL` defaults to `gemini-3.8-flash`. The Blueprint uses Python 3.12, installs pinned production dependencies, starts Uvicorn on Render's `$PORT`, and checks `/health`. Competitor similarity uses local TF-IDF vectors and does not download or load a language model.

For a manual Web Service setup, use:

- **Runtime:** Python 3.12
- **Plan:** Paid always-on service sized for the API workload
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health check path:** `/health`

Add the same provider keys in the Render service's Environment settings. Do not commit or place secret values in `render.yaml`.
