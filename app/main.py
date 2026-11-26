
import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from app.services.website_extractor import extract_website_content
from app.services.llm_service import analyze_with_ai, reset_analysis_id, set_analysis_id
from app.services.competitor_discovery import discover_competitors
from app.utils.errors import ServiceError, WebsiteContentError


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ANALYSIS_TIMEOUT_SECONDS = 120
MAX_CONCURRENT_ANALYSES = 3
_analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)

app = FastAPI(
    title="AI-Powered Website Link Analysis API",
    description="Analyze website content and generate business insights.",
    version="1.0.0"
)


class LinkRequest(BaseModel):
    url: HttpUrl = Field(max_length=2048)


class HealthResponse(BaseModel):
    status: str


class WebsiteSummary(BaseModel):
    url: str
    title: str
    meta_description: str


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    competitor_analysis: dict[str, Any]
    target_search_queries: list[Any]
    seo_analysis: dict[str, Any]
    top_services: list[Any]
    target_audience: dict[str, Any]
    discovered_competitors: list[dict[str, Any]] = Field(default_factory=list)
    competitor_discovery_status: dict[str, Any] | None = None


class LinkAnalysisResponse(BaseModel):
    success: bool
    website: WebsiteSummary
    analysis: AnalysisResponse


def _error_body(code: str, message: str) -> dict:
    return {
        "success": False,
        "error": {"code": code, "message": message},
        # Retain detail for existing API clients from Phase 1.
        "detail": message
    }


def _log_discovery_summary(count: int, reason: str | None = None, error_code: str | None = None) -> None:
    fields: dict[str, Any] = {"discovered_competitor_count": count}
    if reason:
        fields["zero_result_reason"] = reason
    if error_code:
        fields["error_code"] = error_code
    logger.info(
        "competitor_discovery.summary %s",
        json.dumps(fields, sort_keys=True, separators=(",", ":")),
        extra={"diagnostic_event": "competitor_discovery.summary", **fields}
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    error: RequestValidationError
):
    del request
    errors = error.errors()
    url_error = any(
        "url" in tuple(str(part) for part in item.get("loc", ()))
        for item in errors
    )
    message = (
        "Provide a valid HTTP or HTTPS website URL."
        if url_error
        else "The request body is missing or malformed."
    )
    return JSONResponse(
        status_code=422,
        content={
            **_error_body("invalid_request", message),
            "validation_errors": [
                {
                    "location": [str(part) for part in item.get("loc", ())],
                    "message": item.get("msg", "Invalid value.")
                }
                for item in errors
            ]
        }
    )


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, error: ServiceError):
    del request
    return JSONResponse(
        status_code=error.status_code,
        content=_error_body(error.code, error.public_message)
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, error: HTTPException):
    del request
    message = error.detail if isinstance(error.detail, str) else "The request could not be completed."
    return JSONResponse(
        status_code=error.status_code,
        content=_error_body("http_error", message),
        headers=error.headers
    )


@app.get("/health")
def health_check() -> HealthResponse:
    return {"status": "healthy"}


@app.post("/link-analysis", response_model=LinkAnalysisResponse)
async def link_analysis(request: LinkRequest):
    analysis_token = set_analysis_id(uuid.uuid4().hex)
    try:
        async with _analysis_semaphore, asyncio.timeout(ANALYSIS_TIMEOUT_SECONDS):
            # Step 1: Extract and validate website content.
            url = str(request.url)
            website_data = await extract_website_content(url)
            if not isinstance(website_data, dict) or not isinstance(website_data.get("url"), str):
                raise ServiceError(
                    "invalid_extraction_response",
                    "The website extractor returned an incomplete response.",
                    502
                )
            website_data.setdefault("title", "")
            website_data.setdefault("meta_description", "")
            content = website_data.get("content")
            if not isinstance(content, str) or not content.strip():
                raise WebsiteContentError(
                    "The website extractor returned no usable page content."
                )
            if not isinstance(website_data["title"], str) or not isinstance(website_data["meta_description"], str):
                raise ServiceError(
                    "invalid_extraction_response",
                    "The website extractor returned an invalid response.",
                    502
                )

            # Step 2: Generate the core Gemini analysis.
            analysis = await analyze_with_ai(website_data)
            if not isinstance(analysis, dict):
                raise ServiceError(
                    "invalid_analysis_response",
                    "The analysis service returned an invalid response.",
                    502
                )

            # Step 3: Competitor discovery is optional. Preserve the core
            # report if Tavily, Gemini discovery, or ranking is unavailable.
            analysis.setdefault("discovered_competitors", [])
            analysis.setdefault(
                "competitor_discovery_status",
                {"status": "complete", "message": None}
            )
            logger.info(
                    "competitor_discovery.api_started",
                extra={"diagnostic_event": "competitor_discovery.api_started"}
            )
            try:
                discovered = await discover_competitors(
                    analysis,
                    website_data["url"]
                )
                if not isinstance(discovered, list):
                    raise ServiceError(
                        "invalid_discovery_response",
                        "Competitor discovery returned an invalid response.",
                        502
                    )
                analysis["discovered_competitors"] = discovered
                logger.info(
                    "competitor_discovery.api_completed discovered_competitor_count=%d",
                    len(discovered),
                    extra={
                        "diagnostic_event": "competitor_discovery.api_completed",
                        "discovered_competitor_count": len(discovered)
                    }
                )
            except ServiceError as error:
                logger.warning("Optional competitor discovery unavailable (%s).", error.code)
                logger.warning(
                    "competitor_discovery.api_failed",
                    extra={
                        "diagnostic_event": "competitor_discovery.api_failed",
                        "error_type": type(error).__name__,
                        "error_code": error.code
                    }
                )
                _log_discovery_summary(
                    0, reason="discovery_unavailable", error_code=error.code
                )
                analysis["discovered_competitors"] = []
                analysis["competitor_discovery_status"] = {
                    "status": "unavailable",
                    "message": error.public_message
                }
            except Exception as error:
                logger.warning(
                    "Optional competitor discovery failed (%s).",
                    type(error).__name__
                )
                error_code = getattr(error, "code", None) or getattr(error, "status_code", None)
                if error_code is not None:
                    error_code = "".join(
                        character for character in str(error_code)
                        if character.isalnum() or character in "_-"
                    )[:64] or None
                _log_discovery_summary(
                    0, reason="discovery_failed", error_code=error_code
                )
                logger.warning(
                    "competitor_discovery.api_failed",
                    extra={
                        "diagnostic_event": "competitor_discovery.api_failed",
                        "error_type": type(error).__name__,
                        "error_code": error_code
                    }
                )
                analysis["discovered_competitors"] = []
                analysis["competitor_discovery_status"] = {
                    "status": "unavailable",
                    "message": "Competitor discovery could not complete; the website analysis is still available."
                }

            try:
                return LinkAnalysisResponse.model_validate({
                    "success": True,
                    "website": {
                        "url": website_data["url"],
                        "title": website_data["title"],
                        "meta_description": website_data["meta_description"]
                    },
                    "analysis": analysis
                })
            except ValidationError as error:
                raise ServiceError(
                    "invalid_analysis_response",
                    "The analysis service returned an incomplete response.",
                    502
                ) from error

    except TimeoutError as error:
        raise ServiceError(
            "analysis_timeout",
            "The analysis exceeded its time limit. Please try again.",
            504
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ServiceError:
        raise
    except Exception as error:
        logger.error("Website analysis failed (%s).", type(error).__name__)
        raise ServiceError(
            "analysis_failed",
            "Website analysis could not complete. Please try again.",
            502
        ) from error
    finally:
        reset_analysis_id(analysis_token)


# Serve the frontend from the project-level frontend directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="frontend"
)
