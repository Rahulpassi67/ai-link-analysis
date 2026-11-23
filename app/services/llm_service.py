

import asyncio
import json
import logging
import re

from google import genai
from google.genai import types

from app.config import ConfigurationError, GEMINI_API_KEY, GEMINI_MODEL
from app.utils.errors import (
    ExternalServiceError,
    WebsiteContentError,
    classify_external_error,
)

logger = logging.getLogger(__name__)
GEMINI_REQUEST_TIMEOUT = 20


# ============================================================
# 1. INITIALIZE GEMINI CLIENT
# ============================================================

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=GEMINI_REQUEST_TIMEOUT * 1000,
            retry_options=types.HttpRetryOptions(attempts=1)
        )
    )
else:
    client = None


# ============================================================
# 2. WEBSITE DATA VALIDATION
# ============================================================

def validate_website_data(website_data: dict) -> None:
    """
    Validate extracted website content before sending it
    to Gemini.

    Rejects blocked pages, CAPTCHA pages and insufficient
    website content.
    """

    if not isinstance(website_data, dict):
        raise ValueError("Invalid website extraction data.")

    title = str(website_data.get("title", "")).strip()

    description = str(
        website_data.get("meta_description", "")
    ).strip()

    content = str(
        website_data.get("content", "")
    ).strip()

    if not content:
        raise ValueError(
            "No website content was extracted."
        )

    combined = f"{title} {description} {content}".lower()

    blocked_phrases = [
        "your request has been blocked",
        "access denied",
        "verify you are human",
        "verify that you are human",
        "captcha",
        "robot check",
        "security check",
        "unusual traffic",
        "automated requests",
        "request blocked",
        "enable javascript to continue",
        "temporarily blocked",
        "checking your browser",
        "attention required"
    ]

    for phrase in blocked_phrases:
        if phrase in combined:
            raise ValueError(
                "Website extraction was blocked or returned "
                "a security challenge. AI analysis cannot "
                "proceed using this content."
            )

    # Require meaningful extracted website content.
    if len(content) < 300:
        raise ValueError(
            "Insufficient website content extracted. "
            "At least 300 characters of page content "
            "are required for reliable analysis."
        )

    # Reject pages that contain only a generic title.
    if len(title) > 300:
        raise ValueError(
            "Website returned an invalid page title."
        )


# ============================================================
# 3. INFERRED INFORMATION NORMALIZATION
# ============================================================

def normalize_inferred_fields(services: list) -> list:
    """
    The inferred flag describes whether the service itself
    was found in the website content.

    Industry and customer fields can independently be inferred.
    """
    for service in services:
        industry = str(
            service.get("target_industry", "")
        ).strip()

        customer = str(
            service.get("target_customer", "")
        ).strip()

        # Do not change service["inferred"] based on industry
        # or customer labels.
        service["inferred"] = service.get("inferred", False) is True

        if (
            industry
            and "(inferred)" in industry.lower()
            and "(Inferred)" not in industry
        ):
            service["target_industry"] = (
                industry.replace("(inferred)", "(Inferred)")
            )

        if (
            customer
            and "(inferred)" in customer.lower()
            and "(Inferred)" not in customer
        ):
            service["target_customer"] = (
                customer.replace("(inferred)", "(Inferred)")
            )

    return services


# ============================================================
# 4. GENERAL VALIDATION HELPERS
# ============================================================

def validate_required_fields(
    data: dict,
    required_fields: list,
    section_name: str
) -> None:

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid {section_name} format."
        )

    for field in required_fields:

        if field not in data:
            raise ValueError(
                f"Missing {section_name} field: {field}"
            )


def validate_string_list(
    value,
    field_name: str
) -> None:

    if not isinstance(value, list):
        raise ValueError(
            f"{field_name} must be a list."
        )

    for item in value:

        if not isinstance(item, str):
            raise ValueError(
                f"Every item in {field_name} must be a string."
            )


_VALIDATION_SECTION_PATHS = {
    "competitor analysis": "competitor_analysis",
    "competitor": "competitor_analysis.competitors[]",
    "search query": "target_search_queries[]",
    "seo analysis": "seo_analysis",
    "service": "top_services[]",
    "target audience": "target_audience",
}


def _validation_failure_details(error: ValueError) -> tuple[str, list[str], list[str]]:
    """Extract safe schema identifiers without logging model or source text."""
    message = str(error)

    if message.startswith("Missing required sections:"):
        missing = [
            field.strip()
            for field in message.partition(":")[2].split(",")
            if field.strip() in {
                "competitor_analysis",
                "target_search_queries",
                "seo_analysis",
                "top_services",
                "target_audience",
            }
        ]
        return "missing_required_sections", missing, []

    if message.startswith("Gemini must return between 1 and 8 relevant search queries"):
        return "invalid_schema", [], ["target_search_queries"]

    missing_match = re.fullmatch(r"Missing (.+) field: ([A-Za-z_]+)", message)
    if missing_match:
        section = _VALIDATION_SECTION_PATHS.get(missing_match.group(1))
        field = missing_match.group(2)
        if section and re.fullmatch(r"[A-Za-z_]+", field):
            return "missing_field", [f"{section}.{field}"], []

    specific_failures = (
        (r"SEO keyword_intents must be an object\.", "seo_analysis.keyword_intents", "object"),
        (r"SEO keyword_intents must contain all required keyword groups\.", "seo_analysis.keyword_intents", "object with required keyword groups"),
        (r"SEO keyword_intents\.([A-Za-z_]+) must be a list\.", r"seo_analysis.keyword_intents.\1", "array"),
        (r"SEO keyword_intents\.([A-Za-z_]+) entries must be objects\.", r"seo_analysis.keyword_intents.\1[]", "array of objects"),
        (r"SEO keyword_intents\.([A-Za-z_]+)\.keyword must be a non-empty string\.", r"seo_analysis.keyword_intents.\1[].keyword", "non-empty string"),
        (r"SEO keyword_intents\.([A-Za-z_]+)\.intent has an invalid search intent\.", r"seo_analysis.keyword_intents.\1[].intent", "one of: informational, commercial, transactional, navigational"),
        (r"SEO keyword_intents\.([A-Za-z_]+) contains duplicate keywords\.", r"seo_analysis.keyword_intents.\1[]", "unique keyword entries"),
        (r"SEO keyword_intents\.([A-Za-z_]+) is missing an intent for a keyword\.", r"seo_analysis.keyword_intents.\1[]", "intent entry for every keyword"),
        (r"SEO content_gaps must be a list\.", "seo_analysis.content_gaps", "array"),
        (r"Each SEO content gap must be an object\.", "seo_analysis.content_gaps[]", "object"),
        (r"SEO content gap field ([A-Za-z_]+) must be a non-empty string\.", r"seo_analysis.content_gaps[].\1", "non-empty string"),
        (r"SEO prioritized_recommendations must be a list\.", "seo_analysis.prioritized_recommendations", "array"),
        (r"Each prioritized SEO recommendation must be an object\.", "seo_analysis.prioritized_recommendations[]", "object"),
        (r"SEO recommendation field ([A-Za-z_]+) must be a non-empty string\.", r"seo_analysis.prioritized_recommendations[].\1", "non-empty string"),
        (r"SEO recommendation field ([A-Za-z_]+) must be a string\.", r"seo_analysis.prioritized_recommendations[].\1", "string"),
        (r"SEO recommendation priority must be High, Medium or Low\.", "seo_analysis.prioritized_recommendations[].priority", "one of: High, Medium, Low"),
        (r"SEO recommendation implementation_effort must be High, Medium or Low\.", "seo_analysis.prioritized_recommendations[].implementation_effort", "one of: High, Medium, Low"),
        (r"Industry segment evidence must be a list\.", "target_audience.industry_segment_evidence", "array"),
        (r"Industry segment evidence entries must be objects\.", "target_audience.industry_segment_evidence[]", "object"),
        (r"Industry segment evidence fields must be strings\.", "target_audience.industry_segment_evidence[]", "string fields"),
        (r"Audience evidence must be a list\.", "target_audience.audience_evidence", "array"),
        (r"Audience evidence entries must be objects\.", "target_audience.audience_evidence[]", "object"),
        (r"Audience evidence fields must be strings\.", "target_audience.audience_evidence[]", "string fields"),
        (r"Search query platform must be a string\.", "target_search_queries[].platform", "non-empty string"),
        (r"Unsupported search platform\.?", "target_search_queries[].platform", "supported platform name"),
        (r"Search query platform_type is invalid for .+\.", "target_search_queries[].platform_type", "platform type matching the selected platform"),
        (r"Search query intent is unsupported\.", "target_search_queries[].intent", "one of: informational, commercial, transactional, navigational"),
        (r"Search query does not support its .+ intent label\.", "target_search_queries[].intent", "intent supported by query wording"),
        (r"Social platform query lacks a platform-specific content cue\.", "target_search_queries[].query", "social-platform query with a relevant content cue"),
        (r"Video platform query lacks a video-specific cue\.", "target_search_queries[].query", "video-related query wording"),
        (r"Community query lacks a community-specific cue\.", "target_search_queries[].query", "community-related query wording"),
        (r"Marketplace query lacks a marketplace-specific cue\.", "target_search_queries[].query", "marketplace-related query wording"),
        (r"Search query relevance is not supported by its topic and rationale\.", "target_search_queries[].relevance", "relevance aligned with query and business profile"),
        (r"Search query is not relevant to extracted website content\.", "target_search_queries[].query", "query relevant to extracted website content"),
    )
    for pattern, path, expected in specific_failures:
        match = re.fullmatch(pattern, message)
        if match:
            if "\\1" in path:
                path = path.replace("\\1", match.group(1))
            if any(marker in message for marker in (
                "must be a ", "must be an ", "must be non-empty", "entries must be", "fields must be"
            )):
                category = "invalid_field_type"
            elif "must contain all required keyword groups" in message:
                category = "invalid_schema"
            else:
                category = "invalid_field_value"
            return category, [], [path]

    invalid_matchers = (
        (r"Competitor field ([A-Za-z_]+) must be (?:a )?(?:string|list)\.", "competitor_analysis"),
        (r"Search query field ([A-Za-z_]+) must be a non-empty string\.", "target_search_queries[]"),
        (r"SEO field ([A-Za-z_]+) must be a string\.", "seo_analysis"),
        (r"Service field ([A-Za-z_]+) must be a string\.", "top_services[]"),
        (r"Audience field ([A-Za-z_]+) must be a string\.", "target_audience"),
    )
    for pattern, prefix in invalid_matchers:
        match = re.fullmatch(pattern, message)
        if match:
            return "invalid_field_type", [], [f"{prefix}.{match.group(1)}"]

    list_match = re.fullmatch(r"Every item in (.+) must be a string\.", message)
    if list_match:
        field_path = {
            "competitor strengths": "competitor_analysis.competitors[].strengths",
            "competitor weaknesses": "competitor_analysis.competitors[].weaknesses",
            "products_services": "competitor_analysis.products_services[]",
            "brand_advantages": "competitor_analysis.brand_advantages[]",
            "positioning_opportunities": "competitor_analysis.positioning_opportunities[]",
        }.get(list_match.group(1))
        if field_path:
            return "invalid_field_type", [], [field_path]
        for section, fields in (
            ("seo_analysis", {
                "primary_keywords", "secondary_keywords", "long_tail_keywords",
                "business_keywords", "branded_product_keywords", "content_recommendations",
            }),
            ("target_audience", {
                "industry_segments", "business_types", "customer_profiles",
                "geographic_focus", "needs_and_challenges", "interests",
                "recommended_marketing_segments",
            }),
        ):
            if list_match.group(1) in fields:
                return "invalid_field_type", [], [f"{section}.{list_match.group(1)}"]

    fixed_invalid_fields = {
        "Gemini response must be a JSON object.": ["$"],
        "Invalid competitor analysis format.": ["competitor_analysis"],
        "Invalid search query format.": ["target_search_queries[]"],
        "Invalid SEO analysis format.": ["seo_analysis"],
        "Invalid service format.": ["top_services[]"],
        "Invalid target audience format.": ["target_audience"],
        "Each competitor must be an object.": ["competitor_analysis.competitors[]"],
        "Each service must be an object.": ["top_services[]"],
        "Competitor name must be a string.": ["competitor_analysis.competitors[].name"],
        "Service inferred field must be Boolean.": ["top_services[].inferred"],
        "Gemini must return exactly 5 search queries.": ["target_search_queries"],
        "Search queries must cover Google, Facebook, Instagram, LinkedIn and X exactly once.": ["target_search_queries[].platform"],
        "Gemini must return exactly 5 services.": ["top_services"],
    }
    if message in fixed_invalid_fields:
        category = "invalid_root_type" if message == "Gemini response must be a JSON object." else "invalid_schema"
        return category, [], fixed_invalid_fields[message]

    if message.startswith("Unsupported search platform:"):
        return "invalid_field_value", [], ["target_search_queries[].platform"]

    return "schema_constraint_failed", [], []


def _expected_field_types(error: ValueError, invalid_fields: list[str]) -> dict[str, str]:
    """Map validator failures to safe expected-type descriptions."""
    message = str(error)
    expected = None
    message_paths = (
        ("SEO keyword_intents must be an object", "object"),
        ("must contain all required keyword groups", "object with required keyword groups"),
        ("entries must be objects", "array of objects"),
        ("must be an object", "object"),
        ("must be a list", "array"),
        ("fields must be strings", "string fields"),
    )
    for marker, description in message_paths:
        if marker in message:
            expected = description
            break
    if expected is None:
        if "must be a non-empty string" in message or "must be a string" in message:
            expected = "non-empty string" if "non-empty" in message else "string"
        elif "must be Boolean" in message:
            expected = "boolean"
        elif "must be High, Medium or Low" in message:
            expected = "one of: High, Medium, Low"
        elif "invalid search intent" in message or "intent is unsupported" in message:
            expected = "one of: informational, commercial, transactional, navigational"
        elif "exactly 5 services" in message:
            expected = "array containing exactly 5 objects"
        elif "between 1 and 8" in message:
            expected = "array containing 1 to 8 query objects"
        elif "platform_type is invalid" in message:
            expected = "platform type matching the selected platform"
        elif "does not support its" in message:
            expected = "query text supporting its declared intent"
        elif "is not supported by its topic" in message:
            expected = "relevance text supported by both the query and business profile"
        elif "not relevant to extracted website content" in message:
            expected = "query relevant to extracted website content"
        elif "Social platform query lacks" in message:
            expected = "social-platform query with a relevant content cue"
        elif "Video platform query lacks" in message:
            expected = "video-related query wording"
        elif "Community query lacks" in message:
            expected = "community-related query wording"
        elif "Marketplace query lacks" in message:
            expected = "marketplace-related query wording"
        elif "must contain all required keyword groups" in message:
            expected = "object containing primary, secondary, long_tail, business, and branded_products arrays"

    return {field: expected for field in invalid_fields} if expected else {}


def _safe_metadata_value(value) -> str | int | float | bool | None:
    """Convert SDK metadata to a short, content-free scalar for diagnostics."""
    if value is None:
        return value
    enum_name = getattr(value, "name", None)
    if enum_name is None and isinstance(value, (int, float, bool)):
        return value
    label = enum_name or getattr(value, "value", None) or value
    normalized = str(label).rsplit(".", 1)[-1]
    return normalized[:80] if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", normalized) else type(value).__name__


def _gemini_response_diagnostics(response) -> tuple[str | None, dict]:
    """Extract only safe finish/usage metadata, never candidate text."""
    candidates = getattr(response, "candidates", None)
    candidates = candidates if isinstance(candidates, (list, tuple)) else []
    finish_reason = _safe_metadata_value(
        getattr(candidates[0], "finish_reason", None)
    ) if candidates else None
    usage = getattr(response, "usage_metadata", None)
    usage_fields = (
        "prompt_token_count", "candidates_token_count", "total_token_count",
        "thoughts_token_count", "cached_content_token_count",
    )
    usage_summary = {}
    for field in usage_fields:
        value = getattr(usage, field, None) if usage is not None else None
        if isinstance(value, int):
            usage_summary[field] = value
    metadata = {"candidate_count": len(candidates)}
    model_version = _safe_metadata_value(getattr(response, "model_version", None))
    if model_version is not None:
        metadata["model_version"] = model_version
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = _safe_metadata_value(getattr(prompt_feedback, "block_reason", None))
    if block_reason not in (None, "UNSPECIFIED", 0):
        metadata["prompt_block_reason"] = block_reason
    if usage_summary:
        metadata["usage"] = usage_summary
    return (str(finish_reason) if finish_reason is not None else None), metadata


def _top_level_keys(value) -> list[str]:
    if not isinstance(value, dict):
        return []
    # JSON object keys are model-controlled too. Keep ordinary schema names
    # useful for debugging, but do not copy arbitrary key text into logs.
    keys = {
        key if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", key)
        else "<non_schema_key>"
        for key in value
    }
    return sorted(keys)[:32]


def _log_gemini_response_failure(
    *,
    failure_category: str,
    attempt: int,
    missing_fields: list[str] | None = None,
    invalid_fields: list[str] | None = None,
    json_line: int | None = None,
    json_column: int | None = None,
    json_parse_failed: bool = False,
    top_level_keys: list[str] | None = None,
    finish_reason: str | None = None,
    response_metadata: dict | None = None,
    expected_field_types: dict[str, str] | None = None,
) -> None:
    """Emit structured, content-free diagnostics for unusable Gemini output."""
    missing = missing_fields or []
    invalid = invalid_fields or []
    diagnostic = {
        "diagnostic_event": "gemini.response_validation_failed",
        "failure_category": failure_category,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "expected_field_types": expected_field_types or {},
        "top_level_keys": top_level_keys or [],
        "json_parse_failed": json_parse_failed,
        "json_line": json_line,
        "json_column": json_column,
        "finish_reason": finish_reason,
        "response_metadata": response_metadata or {},
        "attempt": attempt,
    }
    logger.warning(
        "gemini.response_validation_failed %s",
        json.dumps(diagnostic, sort_keys=True, separators=(",", ":")),
        extra=diagnostic,
    )


SEO_KEYWORD_GROUPS = {
    "primary": "primary_keywords",
    "secondary": "secondary_keywords",
    "long_tail": "long_tail_keywords",
    "business": "business_keywords",
    "branded_products": "branded_product_keywords",
}
SEO_SEARCH_INTENTS = {
    "informational", "commercial", "transactional", "navigational"
}
SEARCH_PLATFORM_TYPES = {
    "Google": "search_engine",
    "Bing": "search_engine",
    "DuckDuckGo": "search_engine",
    "Yahoo": "search_engine",
    "Facebook": "social_network",
    "Instagram": "social_network",
    "LinkedIn": "social_network",
    "X": "social_network",
    "YouTube": "video_platform",
    "Reddit": "community",
    "Amazon": "marketplace",
}
KEYWORD_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "of", "on", "or", "the", "to", "with", "online", "website", "platform",
    "software", "solution", "solutions", "service", "services", "business",
    "company", "product", "products", "tool", "tools", "best", "top",
}
GENERIC_KEYWORDS = {
    "business", "company", "marketing", "online", "platform",
    "product", "products", "service", "services", "software",
    "solution", "solutions", "technology", "website", "best", "top",
}


def _normalize_keyword(value: str) -> str:
    normalized = re.sub(r"\be[\s-]?commerce\b", " ecommerce ", value.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", normalized))


def _source_text(website_data: dict | None) -> str:
    if not isinstance(website_data, dict):
        return ""
    values = [website_data.get(key, "") for key in ("title", "meta_description", "content")]
    headings = website_data.get("headings", [])
    if isinstance(headings, list):
        values.extend(item for item in headings if isinstance(item, str))
    return " ".join(item for item in values if isinstance(item, str))


def _source_supports_phrase(phrase: str, source_text: str) -> bool:
    normalized_phrase = _normalize_keyword(phrase)
    normalized_source = _normalize_keyword(source_text)
    return bool(normalized_phrase and f" {normalized_phrase} " in f" {normalized_source} ")


def _find_source_quote(phrase: str, source_text: str) -> str | None:
    """Return the exact extracted wording for a directly observed claim."""
    phrase = phrase.strip()
    if not phrase:
        return None
    pattern = re.escape(phrase)
    pattern = pattern.replace(r"\ ", r"\s+")
    match = re.search(rf"(?<!\w){pattern}(?!\w)", source_text, re.IGNORECASE)
    return match.group(0) if match else None


def _keyword_is_branded(keyword: str, brand_name: str, product_names: list[str]) -> bool:
    normalized = _normalize_keyword(keyword)
    normalized_brand = _normalize_keyword(brand_name)
    if normalized_brand and normalized == normalized_brand:
        return True
    if any(
        (product := _normalize_keyword(name))
        and (f" {product} " in f" {normalized} " or f" {normalized} " in f" {product} ")
        for name in product_names
    ):
        return True
    return False


def _keyword_is_relevant(keyword: str, source_text: str, profile_text: str) -> bool:
    if not source_text:
        return True
    keyword_tokens = set(_normalize_keyword(keyword).split()) - KEYWORD_STOP_WORDS
    source_tokens = set(_normalize_keyword(source_text).split())
    profile_tokens = set(_normalize_keyword(profile_text).split()) - KEYWORD_STOP_WORDS
    return bool(keyword_tokens & source_tokens and keyword_tokens & profile_tokens)


def validate_search_queries(
    queries: list,
    competitor_profile: dict,
    website_data: dict | None = None,
) -> list:
    """Validate query labels against platform type and visible topic cues."""
    if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
        raise ValueError("Gemini must return between 1 and 8 relevant search queries.")

    required_fields = (
        "query", "platform", "platform_type", "intent", "relevance", "target_audience"
    )
    brand_name = str(competitor_profile.get("brand_name", ""))
    products = competitor_profile.get("products_services", [])
    products = [item for item in products if isinstance(item, str)] if isinstance(products, list) else []
    profile_text = " ".join([
        brand_name,
        str(competitor_profile.get("category", "")),
        *products,
    ])
    anchor_tokens = set(_normalize_keyword(profile_text).split()) - KEYWORD_STOP_WORDS
    source_text = _source_text(website_data)
    social_cues = {
        "social", "post", "posts", "content", "campaign", "community",
        "creator", "engagement", "audience", "ads", "advertising", "group", "groups",
    }
    intent_markers = {
        "informational": r"\b(?:how|what|why|guide|tutorial|learn|tips|examples|ideas)\b",
        "commercial": r"\b(?:best|compare|comparison|versus|vs|review|reviews|alternative|alternatives)\b",
        "transactional": r"\b(?:buy|purchase|pricing|price|cost|plans?|trial|subscribe|order|checkout|sign\s*up|get\s*started|book\s*a\s*demo)\b",
        "navigational": r"\b(?:login|log\s*in|sign\s*in|official|support|documentation|docs)\b",
    }

    for item in queries:
        # platform_type is an additive response field. Its value is
        # deterministic for supported platforms, so older valid Gemini output
        # can be normalized without weakening checks on supplied values.
        if isinstance(item, dict) and "platform_type" not in item:
            platform = item.get("platform")
            item["platform_type"] = (
                SEARCH_PLATFORM_TYPES.get(platform)
                if isinstance(platform, str)
                else None
            )
        validate_required_fields(item, list(required_fields), "search query")
        for field in required_fields:
            if not isinstance(item[field], str) or not item[field].strip():
                raise ValueError(f"Search query field {field} must be a non-empty string.")

        platform = item["platform"]
        if not isinstance(platform, str):
            raise ValueError("Search query platform must be a string.")
        expected_type = SEARCH_PLATFORM_TYPES.get(platform)
        if not expected_type:
            raise ValueError("Unsupported search platform.")
        if item["platform_type"] != expected_type:
            raise ValueError(f"Search query platform_type is invalid for {platform}.")

        intent = item["intent"]
        if not isinstance(intent, str) or intent not in SEO_SEARCH_INTENTS:
            raise ValueError("Search query intent is unsupported.")
        query = item["query"]
        query_lower = query.casefold()
        if not re.search(intent_markers[intent], query_lower):
            if intent != "navigational" or not _keyword_is_branded(query, brand_name, products):
                raise ValueError(f"Search query does not support its {intent} intent label.")

        if expected_type == "social_network":
            query_tokens = set(_normalize_keyword(query).split())
            if not query_tokens.intersection(social_cues):
                raise ValueError("Social platform query lacks a platform-specific content cue.")
        elif expected_type == "video_platform" and not re.search(r"\b(?:video|videos|channel|youtube)\b", query_lower):
            raise ValueError("Video platform query lacks a video-specific cue.")
        elif expected_type == "community" and not re.search(r"\b(?:community|forum|discussion|reddit)\b", query_lower):
            raise ValueError("Community query lacks a community-specific cue.")
        elif expected_type == "marketplace" and not re.search(r"\b(?:marketplace|amazon|buy|purchase|product)\b", query_lower):
            raise ValueError("Marketplace query lacks a marketplace-specific cue.")

        query_tokens = set(_normalize_keyword(query).split()) - KEYWORD_STOP_WORDS
        overlap = query_tokens & anchor_tokens
        rationale_tokens = set(_normalize_keyword(item["relevance"]).split())
        if not overlap or not (overlap & rationale_tokens):
            raise ValueError("Search query relevance is not supported by its topic and rationale.")
        if source_text and not (query_tokens & set(_normalize_keyword(source_text).split())):
            raise ValueError("Search query is not relevant to extracted website content.")
        item["relevance_level"] = "high" if len(overlap) >= 2 else "medium"

    return queries


def validate_seo_analysis(
    seo: dict,
    *,
    brand_name: str = "",
    business_context: str = "",
    product_names: list[str] | None = None,
    website_data: dict | None = None,
) -> dict:
    """Validate and clean the SEO payload while retaining legacy string fields."""
    if not isinstance(seo, dict):
        raise ValueError("Invalid SEO analysis format.")
    # These fields extend the legacy analysis contract. Missing additive
    # fields mean "no supported suggestion"; supplied values are still
    # validated below and malformed values remain errors.
    seo.setdefault("branded_product_keywords", [])
    seo.setdefault("content_gaps", [])
    seo.setdefault("prioritized_recommendations", [])
    intents_were_provided = "keyword_intents" in seo
    if not intents_were_provided:
        seo["keyword_intents"] = {group: [] for group in SEO_KEYWORD_GROUPS}

    required_fields = [
        "primary_keywords", "secondary_keywords", "long_tail_keywords",
        "business_keywords", "seo_title", "meta_description",
        "content_recommendations", "keyword_intents", "content_gaps",
        "prioritized_recommendations",
    ]
    validate_required_fields(seo, required_fields, "SEO analysis")

    for field in (
        "primary_keywords", "secondary_keywords", "long_tail_keywords",
        "business_keywords", "branded_product_keywords", "content_recommendations",
    ):
        validate_string_list(seo[field], field)
    for field in ("seo_title", "meta_description"):
        if not isinstance(seo[field], str):
            raise ValueError(f"SEO field {field} must be a string.")

    intents = seo["keyword_intents"]
    if not isinstance(intents, dict):
        raise ValueError("SEO keyword_intents must be an object.")
    if set(intents) != set(SEO_KEYWORD_GROUPS):
        missing_groups = set(SEO_KEYWORD_GROUPS) - set(intents)
        unexpected_groups = set(intents) - set(SEO_KEYWORD_GROUPS)
        if unexpected_groups or missing_groups - {"branded_products"}:
            raise ValueError("SEO keyword_intents must contain all required keyword groups.")
        optional_intent_groups = set(missing_groups)
        intents.setdefault("branded_products", [])
    else:
        optional_intent_groups = set()
    if not intents_were_provided:
        optional_intent_groups.update(SEO_KEYWORD_GROUPS)

    intent_maps = {}
    for group, keyword_field in SEO_KEYWORD_GROUPS.items():
        entries = intents[group]
        if not isinstance(entries, list):
            raise ValueError(f"SEO keyword_intents.{group} must be a list.")
        intent_map = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"SEO keyword_intents.{group} entries must be objects.")
            keyword = entry.get("keyword")
            intent = entry.get("intent")
            if not isinstance(keyword, str) or not keyword.strip():
                raise ValueError(f"SEO keyword_intents.{group}.keyword must be a non-empty string.")
            if not isinstance(intent, str) or intent not in SEO_SEARCH_INTENTS:
                raise ValueError(f"SEO keyword_intents.{group}.intent has an invalid search intent.")
            normalized = _normalize_keyword(keyword)
            if normalized in intent_map:
                raise ValueError(f"SEO keyword_intents.{group} contains duplicate keywords.")
            intent_map[normalized] = intent
        intent_maps[keyword_field] = intent_map

    source_text = _source_text(website_data)
    profile_text = " ".join([brand_name, business_context, *(product_names or [])])
    product_names = product_names or []
    transfer_to_branded = {}
    business_field = SEO_KEYWORD_GROUPS["business"]
    branded_field = SEO_KEYWORD_GROUPS["branded_products"]
    generic_business = []
    branded_candidates = list(seo[branded_field])
    for group, keyword_field in SEO_KEYWORD_GROUPS.items():
        if group == "branded_products":
            continue
        for keyword in seo[keyword_field]:
            if _keyword_is_branded(keyword, brand_name, product_names):
                if _source_supports_phrase(keyword, source_text):
                    branded_candidates.append(keyword)
                    transfer_to_branded[_normalize_keyword(keyword)] = intent_maps[keyword_field].get(
                        _normalize_keyword(keyword)
                    )
                # A branded product without extracted-page support is removed.
                continue
            if group == "business":
                generic_business.append(keyword)
    # Branded product names are kept in their own category, not in generic
    # primary, secondary, long-tail or business SEO keyword groups.
    for group in ("primary", "secondary", "long_tail"):
        keyword_field = SEO_KEYWORD_GROUPS[group]
        seo[keyword_field] = [
            keyword for keyword in seo[keyword_field]
            if not _keyword_is_branded(keyword, brand_name, product_names)
        ]
    seo[business_field] = generic_business
    seo[branded_field] = branded_candidates
    for keyword, intent in transfer_to_branded.items():
        if intent is not None:
            intent_maps[branded_field].setdefault(keyword, intent)

    # Keep the legacy keyword arrays while filtering irrelevant, generic-only,
    # and repeated entries. Each retained keyword keeps its validated intent.
    seen_keywords = set()
    cleaned_intents = {group: [] for group in SEO_KEYWORD_GROUPS}
    for group, keyword_field in SEO_KEYWORD_GROUPS.items():
        clean_keywords = []
        for keyword in seo[keyword_field]:
            normalized = _normalize_keyword(keyword)
            if not normalized or normalized in GENERIC_KEYWORDS or normalized in seen_keywords:
                continue
            is_branded = group == "branded_products"
            if is_branded:
                if not _keyword_is_branded(keyword, brand_name, product_names):
                    continue
                if source_text and not _source_supports_phrase(keyword, source_text):
                    continue
            elif source_text and not _keyword_is_relevant(keyword, source_text, profile_text):
                continue
            intent = intent_maps[keyword_field].get(normalized)
            if intent is None:
                if group not in optional_intent_groups:
                    raise ValueError(f"SEO keyword_intents.{group} is missing an intent for a keyword.")
                clean_keywords.append(keyword.strip())
                seen_keywords.add(normalized)
                continue
            seen_keywords.add(normalized)
            clean_keywords.append(keyword.strip())
            cleaned_intents[group].append({"keyword": keyword.strip(), "intent": intent})
        seo[keyword_field] = clean_keywords
    seo["keyword_intents"] = cleaned_intents

    gaps = seo["content_gaps"]
    if not isinstance(gaps, list):
        raise ValueError("SEO content_gaps must be a list.")
    clean_gaps = []
    seen_topics = set()
    for gap in gaps:
        if not isinstance(gap, dict):
            raise ValueError("Each SEO content gap must be an object.")
        for field in ("topic", "why_it_matters", "content_opportunity"):
            if not isinstance(gap.get(field), str) or not gap[field].strip():
                raise ValueError(f"SEO content gap field {field} must be a non-empty string.")
        topic_key = _normalize_keyword(gap["topic"])
        if topic_key and topic_key not in seen_topics:
            clean_gaps.append({
                field: gap[field].strip()
                for field in ("topic", "why_it_matters", "content_opportunity")
            })
            seen_topics.add(topic_key)
    seo["content_gaps"] = clean_gaps

    recommendations = seo["prioritized_recommendations"]
    if not isinstance(recommendations, list):
        raise ValueError("SEO prioritized_recommendations must be a list.")
    clean_recommendations = []
    seen_recommendations = set()
    metadata_assessment = build_metadata_assessment(website_data or {}, seo)
    metadata_states = {
        "title_tag": metadata_assessment["title_tag"]["status"],
        "meta_description": metadata_assessment["meta_description"]["status"],
    }
    gap_topics = {_normalize_keyword(gap["topic"]) for gap in seo["content_gaps"]}
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise ValueError("Each prioritized SEO recommendation must be an object.")
        for field in ("recommendation", "expected_benefit"):
            if not isinstance(recommendation.get(field), str) or not recommendation[field].strip():
                raise ValueError(f"SEO recommendation field {field} must be a non-empty string.")
        for field in ("related_to", "related_topic"):
            if not isinstance(recommendation.get(field), str):
                raise ValueError(f"SEO recommendation field {field} must be a string.")
        priority = recommendation.get("priority")
        effort = recommendation.get("implementation_effort")
        if not isinstance(priority, str) or priority not in {"High", "Medium", "Low"}:
            raise ValueError("SEO recommendation priority must be High, Medium or Low.")
        if not isinstance(effort, str) or effort not in {"High", "Medium", "Low"}:
            raise ValueError("SEO recommendation implementation_effort must be High, Medium or Low.")
        related_to = recommendation["related_to"]
        related_topic = _normalize_keyword(recommendation["related_topic"])
        if related_to in metadata_states:
            metadata_cues = {
                "title_tag": r"\b(?:title|page title|title tag)\b",
                "meta_description": r"\b(?:description|meta description|snippet)\b",
            }
            if (
                metadata_states[related_to] == "within_guideline"
                or related_topic
                or not re.search(metadata_cues[related_to], recommendation["recommendation"], re.I)
            ):
                continue
        elif related_to == "content_gap":
            if not related_topic or related_topic not in gap_topics:
                continue
            topic_tokens = set(related_topic.split()) - KEYWORD_STOP_WORDS
            recommendation_tokens = set(
                _normalize_keyword(recommendation["recommendation"]).split()
            ) - KEYWORD_STOP_WORDS
            if not topic_tokens.intersection(recommendation_tokens):
                continue
        else:
            continue
        key = _normalize_keyword(recommendation["recommendation"])
        if key and key not in seen_recommendations:
            clean_recommendations.append({
                "recommendation": recommendation["recommendation"].strip(),
                "priority": priority,
                "expected_benefit": recommendation["expected_benefit"].strip(),
                "implementation_effort": effort,
                "related_to": related_to,
                "related_topic": recommendation["related_topic"].strip(),
            })
            seen_recommendations.add(key)
    seo["prioritized_recommendations"] = clean_recommendations
    seo["content_recommendations"] = [
        item["recommendation"] for item in clean_recommendations
    ]
    return seo


def build_metadata_assessment(website_data: dict, seo: dict) -> dict:
    """Compare AI suggestions with metadata actually extracted from the page."""
    title = website_data.get("title") if isinstance(website_data.get("title"), str) else ""
    description = website_data.get("meta_description") if isinstance(website_data.get("meta_description"), str) else ""
    title = title.strip()
    description = description.strip()

    def status(value: str, minimum: int, maximum: int) -> str:
        if not value:
            return "missing"
        return "within_guideline" if minimum <= len(value) <= maximum else "weak"

    return {
        "title_tag": {
            "observed": title,
            "status": status(title, 30, 60),
            "character_count": len(title),
            "suggestion": str(seo.get("seo_title") or "").strip(),
        },
        "meta_description": {
            "observed": description,
            "status": status(description, 70, 160),
            "character_count": len(description),
            "suggestion": str(seo.get("meta_description") or "").strip(),
        },
    }


def build_on_page_seo_score(website_data: dict) -> dict:
    """Calculate a transparent heuristic score from extracted on-page signals."""
    title = website_data.get("title") if isinstance(website_data.get("title"), str) else ""
    description = website_data.get("meta_description") if isinstance(website_data.get("meta_description"), str) else ""
    content = website_data.get("content") if isinstance(website_data.get("content"), str) else ""
    headings = website_data.get("headings")
    heading_count = sum(
        isinstance(item, str) and bool(item.strip())
        for item in headings
    ) if isinstance(headings, list) else 0
    word_count = len(re.findall(r"\b[\w'-]+\b", content))

    title_score = 0 if not title.strip() else (25 if 30 <= len(title.strip()) <= 60 else 15)
    description_score = 0 if not description.strip() else (25 if 70 <= len(description.strip()) <= 160 else 15)
    heading_score = 20 if heading_count else 0
    content_score = 30 if word_count >= 500 else 20 if word_count >= 250 else 10 if word_count >= 100 else 0
    components = [
        {"factor": "Title tag", "score": title_score, "max_score": 25,
         "finding": "Missing" if not title.strip() else f"{len(title.strip())} observed characters; guideline range is 30–60."},
        {"factor": "Meta description", "score": description_score, "max_score": 25,
         "finding": "Missing" if not description.strip() else f"{len(description.strip())} observed characters; guideline range is 70–160."},
        {"factor": "Headings", "score": heading_score, "max_score": 20,
         "finding": f"{heading_count} extracted heading(s) observed."},
        {"factor": "Extracted content depth", "score": content_score, "max_score": 30,
         "finding": f"{word_count} words extracted; score bands are under 100, 100–249, 250–499, and 500+."},
    ]
    return {
        "score": sum(item["score"] for item in components),
        "max_score": 100,
        "label": "Heuristic on-page SEO score",
        "methodology": "A deterministic checklist based only on extracted title, meta description, headings, and content length. It is not a Google score or ranking estimate.",
        "components": components,
    }


def filter_supported_industry_segments(
    industries: list,
    evidence_items: list,
    website_data: dict | None,
) -> tuple[list[str], list[dict]]:
    """Keep only industry labels with a source quote from the fetched page."""
    if not isinstance(evidence_items, list):
        raise ValueError("Industry segment evidence must be a list.")
    source_text = _source_text(website_data)
    valid_by_name = {}
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            raise ValueError("Industry segment evidence entries must be objects.")
        industry = evidence.get("industry")
        basis = evidence.get("basis")
        quote = evidence.get("supporting_text")
        reason = evidence.get("reason", "")
        if not all(isinstance(value, str) for value in (industry, basis, quote, reason)):
            raise ValueError("Industry segment evidence fields must be strings.")
        if basis not in {"observed", "inferred"} or not industry.strip() or not quote.strip():
            continue
        if basis == "inferred" and not reason.strip():
            continue
        if not _source_supports_phrase(quote, source_text):
            continue
        if basis == "observed":
            if not _source_supports_phrase(industry, quote):
                continue
        else:
            industry_tokens = set(_normalize_keyword(industry).split()) - KEYWORD_STOP_WORDS
            quote_tokens = set(_normalize_keyword(quote).split()) - KEYWORD_STOP_WORDS
            if not industry_tokens.intersection(quote_tokens):
                continue
        valid_by_name.setdefault(_normalize_keyword(industry), {
            "industry": industry.strip(),
            "basis": basis,
            "supporting_text": quote.strip(),
            "reason": reason.strip(),
        })

    # Older or incomplete model responses may omit provenance objects even
    # when an industry label is stated directly in the fetched page. Restore
    # only those exact observations; inferred labels still need AI evidence.
    for industry in industries:
        if not isinstance(industry, str):
            continue
        key = _normalize_keyword(industry)
        quote = _find_source_quote(industry, source_text)
        if key and quote:
            valid_by_name.setdefault(key, {
                "industry": industry.strip(),
                "basis": "observed",
                "supporting_text": quote,
                "reason": "",
            })

    clean_industries = []
    clean_evidence = []
    seen = set()
    ordered_industries = list(industries)
    ordered_industries.extend(
        item["industry"] for item in valid_by_name.values()
    )
    for industry in ordered_industries:
        if not isinstance(industry, str):
            continue
        key = _normalize_keyword(industry)
        evidence = valid_by_name.get(key)
        if evidence and key not in seen:
            clean_industries.append(evidence["industry"])
            clean_evidence.append(evidence)
            seen.add(key)
    return clean_industries, clean_evidence


def ground_audience_claims(audience: dict, website_data: dict | None) -> dict:
    """Map only audience claims backed by validated website evidence."""
    allowed_fields = {
        "primary_audience", "secondary_audience", "business_types",
        "customer_profiles", "geographic_focus", "needs_and_challenges",
        "interests", "recommended_marketing_segments",
    }
    evidence_items = audience.get("audience_evidence")
    if not isinstance(evidence_items, list):
        raise ValueError("Audience evidence must be a list.")
    source_text = _source_text(website_data)
    valid_evidence = {}
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            raise ValueError("Audience evidence entries must be objects.")
        field = evidence.get("field")
        claim = evidence.get("claim")
        basis = evidence.get("basis")
        quote = evidence.get("supporting_text")
        reason = evidence.get("reason", "")
        if not all(isinstance(value, str) for value in (field, claim, basis, quote, reason)):
            raise ValueError("Audience evidence fields must be strings.")
        if field not in allowed_fields or basis not in {"observed", "inferred"}:
            continue
        if (
            not claim.strip()
            or claim.casefold() == "not available"
            or not quote.strip()
            or not _source_supports_phrase(quote, source_text)
        ):
            continue
        if basis == "observed" and not _source_supports_phrase(claim, quote):
            continue
        if basis == "inferred" and not reason.strip():
            continue
        valid_evidence.setdefault((field, _normalize_keyword(claim)), {
            "field": field,
            "claim": claim.strip(),
            "basis": basis,
            "supporting_text": quote.strip(),
            "reason": reason.strip(),
        })

    # Recover explicit audience wording from the model's existing fields when
    # Gemini omitted the separate evidence object. The quote is copied from
    # extracted page text, so unsupported or paraphrased claims stay filtered.
    for field in allowed_fields:
        value = audience[field]
        claims = value if isinstance(value, list) else [value]
        for claim in claims:
            if not isinstance(claim, str) or not claim.strip():
                continue
            key = (field, _normalize_keyword(claim))
            if key in valid_evidence:
                continue
            quote = _find_source_quote(claim, source_text)
            if quote and claim.casefold() != "not available":
                valid_evidence[key] = {
                    "field": field,
                    "claim": claim.strip(),
                    "basis": "observed",
                    "supporting_text": quote,
                    "reason": "",
                }

    for field in allowed_fields:
        value = audience[field]
        evidence_claims = [
            evidence["claim"]
            for (evidence_field, _), evidence in valid_evidence.items()
            if evidence_field == field
        ]
        if isinstance(value, list):
            supported_claims = [
                claim for claim in value
                if (field, _normalize_keyword(claim)) in valid_evidence
            ]
            seen_claims = {_normalize_keyword(claim) for claim in supported_claims}
            for claim in evidence_claims:
                normalized_claim = _normalize_keyword(claim)
                if normalized_claim not in seen_claims:
                    supported_claims.append(claim)
                    seen_claims.add(normalized_claim)
            audience[field] = supported_claims
        elif value.strip() and value.casefold() != "not available":
            key = (field, _normalize_keyword(value))
            if key not in valid_evidence:
                audience[field] = evidence_claims[0] if evidence_claims else "Not available"
        elif evidence_claims:
            audience[field] = evidence_claims[0]
        else:
            audience[field] = "Not available"
    audience["audience_evidence"] = list(valid_evidence.values())
    return audience


# ============================================================
# 5. GEMINI RESPONSE VALIDATION
# ============================================================

def validate_analysis(result: dict, website_data: dict | None = None) -> dict:
    """
    Validate Gemini's JSON response and all required sections.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Gemini response must be a JSON object."
        )

    required_sections = {
        "competitor_analysis",
        "target_search_queries",
        "seo_analysis",
        "top_services",
        "target_audience"
    }

    missing = required_sections - result.keys()

    if missing:
        raise ValueError(
            f"Missing required sections: "
            f"{', '.join(sorted(missing))}"
        )

    # --------------------------------------------------------
    # COMPETITOR ANALYSIS
    # --------------------------------------------------------

    competitor = result["competitor_analysis"]

    required_competitor_fields = [
        "brand_name",
        "overview",
        "category",
        "products_services",
        "competitors",
        "brand_advantages",
        "positioning_opportunities"
    ]

    validate_required_fields(
        competitor,
        required_competitor_fields,
        "competitor analysis"
    )

    for field in [
        "brand_name",
        "overview",
        "category"
    ]:
        if not isinstance(competitor[field], str):
            raise ValueError(
                f"Competitor field {field} must be a string."
            )

    for field in [
        "products_services",
        "competitors",
        "brand_advantages",
        "positioning_opportunities"
    ]:
        if not isinstance(competitor[field], list):
            raise ValueError(
                f"Competitor field {field} must be a list."
            )

    for item in competitor["competitors"]:

        if not isinstance(item, dict):
            raise ValueError(
                "Each competitor must be an object."
            )

        validate_required_fields(
            item,
            ["name", "strengths", "weaknesses"],
            "competitor"
        )

        if not isinstance(item["name"], str):
            raise ValueError(
                "Competitor name must be a string."
            )

        validate_string_list(
            item["strengths"],
            "competitor strengths"
        )

        validate_string_list(
            item["weaknesses"],
            "competitor weaknesses"
        )

    # --------------------------------------------------------
    # TARGET SEARCH QUERIES
    # --------------------------------------------------------

    validate_search_queries(
        result["target_search_queries"],
        competitor,
        website_data,
    )

    # --------------------------------------------------------
    # SEO ANALYSIS
    # --------------------------------------------------------

    seo = result["seo_analysis"]

    validate_seo_analysis(
        seo,
        brand_name=competitor.get("brand_name", ""),
        business_context=" ".join([
            str(competitor.get("category", "")),
            str(competitor.get("overview", "")),
        ]),
        product_names=competitor.get("products_services", []),
        website_data=website_data,
    )

    # --------------------------------------------------------
    # TOP SERVICES
    # --------------------------------------------------------

    services = result["top_services"]

    if not isinstance(services, list) or len(services) != 5:
        raise ValueError(
            "Gemini must return exactly 5 services."
        )

    required_service_fields = [
        "name",
        "description",
        "target_industry",
        "target_customer",
        "business_benefit",
        "inferred"
    ]

    for service in services:

        if not isinstance(service, dict):
            raise ValueError(
                "Each service must be an object."
            )

        if not service.get("target_customer"):
            service["target_customer"] = "Not available"

        if "inferred" not in service:
            service["inferred"] = False

        validate_required_fields(
            service,
            required_service_fields,
            "service"
        )

        for field in required_service_fields[:-1]:

            if not isinstance(service[field], str):
                raise ValueError(
                    f"Service field {field} must be a string."
                )

        if not isinstance(service["inferred"], bool):
            raise ValueError(
                "Service inferred field must be Boolean."
            )

    result["top_services"] = normalize_inferred_fields(
        services
    )

    # --------------------------------------------------------
    # TARGET AUDIENCE
    # --------------------------------------------------------

    audience = result["target_audience"]
    if isinstance(audience, dict):
        # Evidence arrays are additive provenance fields. Without them, keep
        # no unsupported audience or industry claims rather than failing the
        # entire legacy analysis response.
        audience.setdefault("industry_segment_evidence", [])
        audience.setdefault("audience_evidence", [])

    required_audience_fields = [
        "primary_audience",
        "secondary_audience",
        "industry_segments",
        "business_types",
        "customer_profiles",
        "geographic_focus",
        "needs_and_challenges",
        "interests",
        "recommended_marketing_segments",
        "industry_segment_evidence",
        "audience_evidence",
    ]

    validate_required_fields(
        audience,
        required_audience_fields,
        "target audience"
    )

    for field in [
        "primary_audience",
        "secondary_audience"
    ]:
        if not isinstance(audience[field], str):
            raise ValueError(
                f"Audience field {field} must be a string."
            )

    for field in [
        "industry_segments",
        "business_types",
        "customer_profiles",
        "geographic_focus",
        "needs_and_challenges",
        "interests",
        "recommended_marketing_segments"
    ]:
        validate_string_list(
            audience[field],
            field
        )

    audience["industry_segments"], audience["industry_segment_evidence"] = (
        filter_supported_industry_segments(
            audience["industry_segments"],
            audience["industry_segment_evidence"],
            website_data,
        )
    )
    ground_audience_claims(audience, website_data)
    supported_audiences = {
        _normalize_keyword(value)
        for field in ("primary_audience", "secondary_audience", "customer_profiles")
        for value in (
            audience.get(field, [])
            if isinstance(audience.get(field), list)
            else [audience.get(field, "")]
        )
        if isinstance(value, str) and value.strip()
        and value.casefold() != "not available"
    }
    for query in result["target_search_queries"]:
        audience_label = query["target_audience"]
        if audience_label.casefold() != "not available" and _normalize_keyword(audience_label) not in supported_audiences:
            query["target_audience"] = "Not available"

    return result


# ============================================================
# 6. MAIN GEMINI ANALYSIS FUNCTION
# ============================================================

def add_semantic_competitor_ranking(analysis: dict) -> dict:
    """
    Rank Gemini-generated competitor suggestions by semantic
    relevance to the analyzed brand.

    If the ML matcher fails, preserve the original Gemini output.
    """
    competitors = analysis.get(
        "competitor_analysis", {}
    ).get("competitors", [])

    if not competitors:
        return analysis

    try:
        # Import lazily so an ML issue cannot prevent API startup.
        from app.services.semantic_matcher import rank_similar_items

        brand = analysis["competitor_analysis"]

        query_parts = [
            brand.get("brand_name", ""),
            brand.get("category", ""),
            brand.get("overview", ""),
        ]

        query_parts.extend(
            item for item in brand.get("products_services", [])
            if isinstance(item, str)
        )

        query = " ".join(query_parts).strip()

        if not query:
            return analysis

        candidate_texts = []
        candidate_lookup = {}

        for competitor in competitors:
            description = (
                f"{competitor.get('name', '')}. "
                f"Strengths: "
                f"{', '.join(competitor.get('strengths', []))}. "
                f"Weaknesses: "
                f"{', '.join(competitor.get('weaknesses', []))}"
            )

            candidate_texts.append(description)
            candidate_lookup[description] = competitor

        ranked = rank_similar_items(query, candidate_texts)

        ranked_competitors = []

        for item in ranked:
            description = item.get("description")
            original = candidate_lookup.get(description)

            if original is None:
                continue

            competitor = dict(original)
            competitor["similarity_score"] = round(
                float(item["similarity_score"]), 4
            )

            ranked_competitors.append(competitor)

        if ranked_competitors:
            brand["competitors"] = ranked_competitors

    except Exception as error:
        # Never break the working Gemini analysis if ML ranking fails.
        logger.warning("Semantic competitor ranking failed (%s).", type(error).__name__)

    return analysis

async def analyze_with_ai(website_data: dict) -> dict:
    """
    Analyze extracted website information using Google Gemini.

    Generates:
    - Competitor analysis
    - Target search queries
    - SEO analysis
    - Top services
    - Target audience

    Rejects blocked or insufficient website content
    before calling Gemini.
    """

    if client is None:
        raise ConfigurationError(
            "GEMINI_API_KEY is not configured. "
            "Set it in the environment or .env file."
        )

    # Validate website content BEFORE AI analysis.
    try:
        validate_website_data(website_data)
    except ValueError as error:
        raise WebsiteContentError(str(error)) from error

    prompt = f"""
You are an expert business analyst, competitor researcher,
SEO strategist and digital marketing consultant.

Analyze the following extracted website information.

WEBSITE DATA:

{json.dumps(website_data, ensure_ascii=False)}


IMPORTANT SOURCE RULES:

1. Use the extracted website content as the primary source.

2. Identify actual products and services explicitly mentioned
   in the website content.

3. For a product or service explicitly named or described
   on the website, set inferred=false.

4. Set inferred=true ONLY when the service itself is an
   AI-generated suggestion and is not explicitly supported
   by the website content.

5. The inferred Boolean describes the existence of the
   service, NOT whether its target industry or customer
   segment is inferred.

6. If the target industry or customer segment is inferred,
   append "(Inferred)" to that individual field.

7. Do not invent products, services, features or competitors.

8. Return exactly five top_services. If fewer than five
   actual services are identified, fill remaining entries
   with clearly labelled AI-generated suggestions and
   set inferred=true for those entries.

9. Include competitor names only when they are reasonably
   known and relevant. Do not claim they were verified
   from the supplied website.

10. If competitor information is unavailable, return an
    empty competitors array [].

11. Clearly distinguish website claims from independently
    verified facts. Attribute promotional claims to the brand.

12. Never invent market share, revenue, customer counts,
    rankings, search volumes or numerical statistics.

13. Recommendations may be AI-generated but must be
    relevant to the actual extracted website information.

14. Do not treat security pages, errors or generic notices
    as business information.

15. Use "Not available" where a required text field cannot
    be supported.


Return ONLY valid JSON with exactly these five top-level sections:

{{
  "competitor_analysis": {{
    "brand_name": "",
    "overview": "",
    "category": "",
    "products_services": [],
    "competitors": [
      {{
        "name": "",
        "strengths": [],
        "weaknesses": []
      }}
    ],
    "brand_advantages": [],
    "positioning_opportunities": []
  }},

  "target_search_queries": [
    {{
      "query": "",
      "platform": "",
      "platform_type": "search_engine",
      "intent": "",
      "relevance": "",
      "target_audience": ""
    }}
  ],

  "seo_analysis": {{
    "primary_keywords": [],
    "secondary_keywords": [],
    "long_tail_keywords": [],
    "business_keywords": [],
    "branded_product_keywords": [],
    "seo_title": "",
    "meta_description": "",
    "content_recommendations": [],
    "keyword_intents": {{
      "primary": [],
      "secondary": [],
      "long_tail": [],
      "business": [],
      "branded_products": []
    }},
    "content_gaps": [],
    "prioritized_recommendations": []
  }},

  "top_services": [
    {{
      "name": "",
      "description": "",
      "target_industry": "",
      "target_customer": "",
      "business_benefit": "",
      "inferred": false
    }}
  ],

  "target_audience": {{
    "primary_audience": "",
    "secondary_audience": "",
    "industry_segments": [],
    "industry_segment_evidence": [],
    "business_types": [],
    "customer_profiles": [],
    "geographic_focus": [],
    "needs_and_challenges": [],
    "interests": [],
    "recommended_marketing_segments": [],
    "audience_evidence": []
  }}
}}

STRICT OUTPUT RULES:

1. Return between 1 and 8 target_search_queries, only when a useful
   query is supported by the extracted business profile.

2. Return exactly 5 top_services.

3. Distinguish search engines from social networks. Google, Bing,
   DuckDuckGo and Yahoo are search_engine platforms. Facebook,
   Instagram, LinkedIn and X are social_network platforms. YouTube
   is a video_platform; Reddit is a community; Amazon is a marketplace.
   Set platform_type to the matching type. Never label a social network
   as a search engine and do not force one query per platform.

4. Include relevant competitors only when reasonably
   supported by available information.

5. Do not invent search volumes, rankings, percentages,
   customer counts, revenue figures or factual statistics.

6. Clearly distinguish extracted facts from AI suggestions.

7. Mark inferred services with inferred=true.

8. When industry or customer segments are inferred,
   append "(Inferred)" to their respective fields.

9. If geographic information is unavailable,
   return "Not available".

10. Use specific geographic descriptions when supported.

11. Do not use redundant phrases such as "Global Worldwide".

12. Use "Worldwide" only when supported by the website.

13. Do not repeat irrelevant website content.

14. Keep recommendations practical and specific to the brand.

15. Ensure every required JSON field is present.

16. Return valid JSON only.

17. Do not include Markdown code fences, introductory text,
    explanations or comments outside the JSON object.

18. Competitor strengths and weaknesses must be described
    as AI-generated assessments unless supported by evidence.

19. Do not invent competitor pricing, market share,
    rankings or customer numbers.

20. Never claim that AI-generated recommendations are
    verified search or market research results.

21. Keep SEO keywords specific to the website's products,
    audience and extracted content. Exclude generic, irrelevant
    and repeated keywords across all keyword groups.

22. Provide one search intent for each keyword using only
    informational, commercial, transactional or navigational.

23. The seo_title and meta_description fields are proposed
    replacements, not observations of current page metadata.

24. Suggest content gaps only when the extracted page gives
    enough context; describe them as AI suggestions, not as a
    complete audit of the website.

25. For each prioritized recommendation, use High, Medium or
    Low for both priority and implementation_effort. Explain
    benefits qualitatively; do not invent traffic or ranking gains.

26. Never invent keyword volume, keyword difficulty, traffic,
    Google rankings or search performance measurements.

27. Use empty arrays when the available page content does not
    support relevant keyword or content-gap suggestions.

28. Each keyword_intents item must be an object with "keyword"
    and "intent" strings, and must correspond to a keyword in
    its matching keyword array. Put branded product names in
    branded_product_keywords, not business_keywords. Use that group
    only for named products visible in the extracted page or listed
    as website products/services.

29. Each content_gaps item must contain non-empty "topic",
    "why_it_matters" and "content_opportunity" strings.

30. Each prioritized_recommendations item must contain
    "recommendation", "priority", "expected_benefit",
    "implementation_effort", "related_to" and "related_topic".
    related_to must be title_tag, meta_description or content_gap.
    A content_gap recommendation must reference an exact topic from
    content_gaps; metadata recommendations must address a missing or
    weak observed metadata field. Mirror these recommendations in the
    legacy content_recommendations string array.

31. Every target search query must contain a clear intent cue in the
    query itself: informational (how, what, guide, tips), commercial
    (best, compare, reviews, alternatives), transactional (pricing,
    buy, plans, trial), or navigational (brand/product plus login,
    support, official or documentation). The relevance explanation must
    repeat a specific category, product or service topic that appears in
    both the query and the competitor business profile, then explain why
    the query matches that topic. Do not use a generic statement such as
    "relevant to this business" as the explanation.

32. Use Facebook, Instagram, LinkedIn or X only for queries that
    explicitly concern social content, posts, campaigns, communities,
    creators, engagement, groups or ads. Use platform-specific query
    cues for YouTube, Reddit and Amazon as well.

33. Industry segments must each include an industry_segment_evidence
    object with industry, basis (observed or inferred), supporting_text
    quoted from extracted website text, and reason. Do not emit an
    industry without a source quote; observed labels must appear in it.

34. Ground every target-audience claim with an audience_evidence item
    containing field, claim, basis (observed or inferred), supporting_text
    quoted from the extracted page, and reason. Mark interpretations as
    inferred and do not emit unsupported generic audience claims.

35. Industry and audience evidence supporting_text must be copied from
    the supplied extracted website text; do not invent or rewrite quotes.

36. For target queries, platform_type must use search_engine,
    social_network, video_platform, community or marketplace according
    to platform. Relevance explanations must refer to a business-profile
    topic that appears in the query; do not force social-platform queries.

37. Keep branded product names such as named payment, capital or market
    products in branded_product_keywords. business_keywords should contain
    non-branded concepts supported by the website.
"""

    # ========================================================
    # 7. GEMINI API WITH RETRIES
    # ========================================================

    max_attempts = 3
    schema_correction_used = False

    for attempt in range(max_attempts):

        try:

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                ),
                timeout=GEMINI_REQUEST_TIMEOUT + 3
            )

            finish_reason, response_metadata = _gemini_response_diagnostics(response)

            if not response.text:
                truncated = finish_reason in {"MAX_TOKENS", "LENGTH"}
                _log_gemini_response_failure(
                    failure_category="truncated_output" if truncated else "empty_response",
                    attempt=attempt + 1,
                    finish_reason=finish_reason,
                    response_metadata=response_metadata,
                )
                raise ExternalServiceError(
                    "gemini_incomplete_output" if truncated else "gemini_empty_response",
                    "Gemini output was truncated before a complete analysis could be produced. Please try again."
                    if truncated else "Gemini returned an empty analysis response.",
                    502
                )

            response_text = response.text.strip()

            # Remove accidental Markdown code fences.
            response_text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                response_text,
                flags=re.IGNORECASE
            ).strip()


            # Parse JSON.
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as error:
                _log_gemini_response_failure(
                    failure_category="malformed_json",
                    attempt=attempt + 1,
                    json_line=error.lineno,
                    json_column=error.colno,
                    json_parse_failed=True,
                    finish_reason=finish_reason,
                    response_metadata=response_metadata,
                )
                if (
                    attempt < max_attempts - 1
                    and finish_reason not in {"MAX_TOKENS", "LENGTH"}
                ):
                    logger.warning(
                        "Gemini returned malformed JSON (%s) on attempt %s/%s; retrying.",
                        type(error).__name__,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(min(0.5 * (2 ** attempt), 1.5))
                    continue
                raise ExternalServiceError(
                    "gemini_incomplete_output" if finish_reason in {"MAX_TOKENS", "LENGTH"} else "gemini_invalid_response",
                    "Gemini output was truncated before a complete analysis could be produced. Please try again."
                    if finish_reason in {"MAX_TOKENS", "LENGTH"}
                    else "Gemini returned an invalid analysis response. Please try again.",
                    502
                ) from error

            # Validate all sections and fields.
            try:
                validated_result = validate_analysis(result, website_data=website_data)
            except ValueError as error:
                failure_category, missing_fields, invalid_fields = (
                    _validation_failure_details(error)
                )
                expected_field_types = _expected_field_types(error, invalid_fields)
                _log_gemini_response_failure(
                    failure_category=failure_category,
                    attempt=attempt + 1,
                    missing_fields=missing_fields,
                    invalid_fields=invalid_fields,
                    expected_field_types=expected_field_types,
                    top_level_keys=_top_level_keys(result),
                    finish_reason=finish_reason,
                    response_metadata=response_metadata,
                )
                if (
                    not schema_correction_used
                    and attempt < max_attempts - 1
                    and finish_reason not in {"MAX_TOKENS", "LENGTH"}
                    and (missing_fields or invalid_fields)
                ):
                    schema_correction_used = True
                    correction = {
                        "missing_fields": missing_fields,
                        "invalid_fields": [
                            {
                                "field": field,
                                "expected": expected_field_types.get(field, "valid schema value"),
                            }
                            for field in invalid_fields
                        ],
                    }
                    relevance_correction = ""
                    if "target_search_queries[].relevance" in invalid_fields:
                        relevance_correction = (
                            " For every query with an invalid relevance field, rewrite that field "
                            "to repeat a concrete category, product, or service topic present in "
                            "both that query and competitor_analysis.category or "
                            "competitor_analysis.products_services, and explain their connection. "
                            "Do not rely on a generic relevance statement, brand name alone, or "
                            "platform name alone. Preserve all other query and analysis values."
                        )
                    prompt += (
                        "\n\nCORRECTIVE RESPONSE REQUIREMENT: Your previous JSON failed "
                        "schema validation. Return a complete corrected JSON object. "
                        "Supply every listed missing field and correct every listed invalid field "
                        "to its expected type/value. Preserve all other required analysis fields. "
                        "Do not add commentary or Markdown."
                        + relevance_correction
                        + "\n"
                        + json.dumps(correction, sort_keys=True, separators=(",", ":"))
                    )
                    logger.warning(
                        "Gemini analysis schema validation failed (%s); retrying once with field-level correction.",
                        failure_category,
                    )
                    continue
                raise ExternalServiceError(
                    "gemini_invalid_response",
                    "Gemini returned an incomplete analysis response. Please try again.",
                    502
                ) from error

            # Add ML-based competitor ranking.
            validated_result["seo_analysis"]["metadata_assessment"] = (
                build_metadata_assessment(website_data, validated_result["seo_analysis"])
            )
            validated_result["seo_analysis"]["seo_score"] = build_on_page_seo_score(
                website_data
            )
            return add_semantic_competitor_ranking(validated_result)

        except Exception as error:

            error_message = str(error).upper()

            temporary_error = any(
                term in error_message
                for term in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                    "TIMEOUT",
                    "DEADLINE_EXCEEDED"
                ]
            ) or isinstance(error, TimeoutError)

            if (
                temporary_error
                and attempt < max_attempts - 1
            ):

                logger.warning(
                    "Gemini request failed with a retryable error (%s); retry %s/%s.",
                    type(error).__name__,
                    attempt + 1,
                    max_attempts - 1
                )
                await asyncio.sleep(min(0.5 * (2 ** attempt), 1.5))

                continue

            if isinstance(error, ExternalServiceError):
                raise
            raise classify_external_error("Gemini", error) from error

    raise ExternalServiceError(
        "gemini_unavailable",
        "Gemini could not complete the analysis after multiple attempts.",
        503
    )
