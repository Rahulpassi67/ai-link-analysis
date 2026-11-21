
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from app.services.website_extractor import extract_website_content
from app.services.llm_service import analyze_with_ai


app = FastAPI(
    title="AI-Powered Website Link Analysis API",
    description="Analyze website content and generate business insights.",
    version="1.0.0"
)


class LinkRequest(BaseModel):
    url: HttpUrl
3

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/link-analysis")
async def link_analysis(request: LinkRequest):
    try:
        url = str(request.url)

        website_data = await extract_website_content(url)
        analysis = await analyze_with_ai(website_data)

        return {
            "success": True,
            "website": {
                "url": website_data["url"],
                "title": website_data["title"],
                "meta_description": website_data["meta_description"]
            },
            "analysis": analysis
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception as error:
        print("ACTUAL ERROR:", repr(error))
        raise HTTPException(
            status_code=502,
            detail="Website analysis failed. Please try again."
        )


# Serve the frontend from the project-level frontend directory.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)

