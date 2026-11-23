
from app.services.evidence_verifier import (
    deduplicate_competitor_excerpts,
    verify_competitors
)
import asyncio
import json
import logging
import re
from urllib.parse import urlparse

from tavily import TavilyClient

from app.config import require_api_key
from app.services.llm_service import client as gemini_client, GEMINI_MODEL
from app.utils.company_names import (
    company_name_matches_text,
    normalize_company_name,
)
from app.utils.errors import ExternalServiceError, classify_external_error

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TAVILY_REQUEST_TIMEOUT = 12
GEMINI_DISCOVERY_TIMEOUT = 20


def _safe_error_code(error: Exception) -> str | None:
    """Return a short structured code without logging arbitrary error text."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if code is None:
        return None
    safe_code = re.sub(r"[^A-Za-z0-9_-]", "", str(code))[:64]
    return safe_code or None


def _log_diagnostic(event: str, level: str = "info", **fields) -> None:
    details = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    getattr(logger, level)(
        "competitor_discovery.%s %s",
        event,
        details,
        extra={"diagnostic_event": f"competitor_discovery.{event}", **fields}
    )


def _log_discovery_summary(
    discovered_count: int,
    reason: str | None = None,
    error_code: str | None = None
) -> None:
    fields = {"discovered_competitor_count": discovered_count}
    if reason:
        fields["zero_result_reason"] = reason
    if error_code:
        fields["error_code"] = error_code
    _log_diagnostic("summary", **fields)


EXCLUDED_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "forbes.com",
    "finance.yahoo.com",
    "techcrunch.com",
    "siliconangle.com",
    "betakit.com",
    "medium.com",
    "reddit.com",
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "tavily.com",
    "g2.com",
    "capterra.com",
    "cbinsights.com",
    "crunchbase.com",
    "pitchbook.com",
    "tracxn.com",
    "glassdoor.com",
    "zoominfo.com",
    "bloomberg.com",
    "wikipedia.org",
}


def normalize_domain(url: str) -> str:
    """Extract a normalized hostname from a URL."""
    try:
        hostname = urlparse(url).hostname or ""
        return hostname.lower().removeprefix("www.")
    except Exception:
        return ""


def root_url(url: str) -> str:
    """Return the HTTPS homepage URL for a domain."""
    domain = normalize_domain(url)
    return f"https://{domain}" if domain else ""


def is_excluded_domain(domain: str) -> bool:
    """Check whether a domain belongs to an excluded source."""
    return any(
        domain == excluded or domain.endswith("." + excluded)
        for excluded in EXCLUDED_DOMAINS
    )


def canonicalize_url(url: str) -> str:
    """Normalize URLs for evidence matching."""
    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return ""

        if parsed.username or parsed.password:
            return ""

        host = parsed.hostname.lower().removeprefix("www.")
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
        if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
            host = f"{host}:{port}"

        return (
            f"{parsed.scheme.lower()}://"
            f"{host}"
            f"{parsed.path.rstrip('/')}"
        )
    except Exception:
        return ""


def build_search_queries(profile: dict) -> list[str]:
    """Build category-focused competitor discovery queries."""

    category = str(profile.get("category", "")).strip()
    products = profile.get("products_services", [])

    if isinstance(products, list):
        product_text = " ".join(
            str(item) for item in products[:3]
        )
    else:
        product_text = str(products)

    queries = []

    if category:
        queries.append(
            f"leading companies and software platforms "
            f"in the {category} market competitors"
        )

    if category and product_text:
        queries.append(
            f"companies offering {product_text} "
            f"in the {category} industry"
        )

    return queries[:2]


def search_tavily(query: str, api_key: str) -> list[dict]:
    """Search the web using Tavily."""

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=10,
        include_answer=False,
        timeout=TAVILY_REQUEST_TIMEOUT
    )

    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ExternalServiceError(
            "tavily_invalid_response",
            "Tavily returned an unexpected search response.",
            502
        )
    return [item for item in response["results"] if isinstance(item, dict)]


async def extract_companies_with_gemini(
    profile: dict,
    search_results: list[dict]
) -> list[dict]:
    """Extract potential competing companies from search evidence."""

    _log_diagnostic("gemini_extraction_started", search_result_count=len(search_results))
    if not gemini_client:
        logger.warning("Gemini client is not configured.")
        _log_diagnostic(
            "gemini_extraction_failed", level="warning",
            error_type="ClientNotConfigured", error_code="gemini_unavailable"
        )
        raise ExternalServiceError(
            "gemini_unavailable",
            "Gemini is unavailable for competitor extraction.",
            503
        )

    evidence = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", "")[:1500]
        }
        for result in search_results
    ]

    prompt = f"""
You are a business competitor research assistant.

Identify actual companies that offer products or services
competing with the analyzed brand.

BRAND:
{profile.get("brand_name", "")}

CATEGORY:
{profile.get("category", "")}

PRODUCTS AND SERVICES:
{json.dumps(profile.get("products_services", []))}

SEARCH EVIDENCE:
{json.dumps(evidence, ensure_ascii=False)}

RULES:
1. Return actual operating companies or software products.
2. Exclude publishers, blogs, news websites, directories,
   agencies and research organizations.
3. Exclude the analyzed brand itself.
4. Do not invent company names, products or facts.
5. Use only information supported by the supplied evidence.
6. Do not invent official website URLs.
7. Return only evidence URLs supplied in the search results.
8. Return at most 8 companies.
9. If evidence is insufficient, return an empty list.
10. Return "unclear" when the supplied evidence does not establish
    whether the company is a direct or indirect competitor.
11. Do not infer direct/indirect status solely from category overlap.
12. Identify a product only when its name appears in the supplied evidence.
13. Keep descriptions grounded in source text and do not invent reasons.
14. Do not claim a competitor relationship based only on appearing
    in the same industry report.

Return valid JSON only in this format:

{{
  "companies": [
    {{
      "name": "Company name",
      "product_name": "Product name",
      "description": "Description supported by evidence",
      "competitor_type": "unclear",
      "competitor_type_reason": "Why the evidence supports this label, or why it is unclear",
      "evidence_urls": [
        "https://source.example"
      ]
    }}
  ]
}}
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=GEMINI_MODEL,
                    contents=prompt
                ),
                timeout=GEMINI_DISCOVERY_TIMEOUT
            )

            if not response.text:
                raise ExternalServiceError(
                    "gemini_discovery_empty_response",
                    "Gemini returned an empty competitor-discovery response.",
                    502
                )

            response_text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                response.text.strip(),
                flags=re.IGNORECASE
            ).strip()

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as error:
                raise ExternalServiceError(
                    "gemini_discovery_invalid_response",
                    "Gemini returned an invalid competitor-discovery response.",
                    502
                ) from error

            if not isinstance(data, dict):
                raise ExternalServiceError(
                    "gemini_discovery_invalid_response",
                    "Gemini returned an unexpected competitor-discovery response.",
                    502
                )
            companies = data.get("companies", [])

            if not isinstance(companies, list):
                raise ExternalServiceError(
                    "gemini_discovery_invalid_response",
                    "Gemini returned an unexpected competitor-discovery response.",
                    502
                )

            _log_diagnostic(
                "gemini_extraction_succeeded",
                company_candidate_count=sum(isinstance(item, dict) for item in companies[:8])
            )
            return [
                company for company in companies
                if isinstance(company, dict)
            ][:8]

        except Exception as exc:
            if isinstance(exc, ExternalServiceError):
                logger.warning(
                    "Gemini company extraction failed (%s) on attempt %s/%s.",
                    type(exc).__name__,
                    attempt + 1,
                    max_attempts
                )
                _log_diagnostic(
                    "gemini_extraction_failed", level="warning",
                    error_type=type(exc).__name__, error_code=_safe_error_code(exc),
                    attempt=attempt + 1, max_attempts=max_attempts
                )
                raise

            error_message = str(exc).upper()
            error_code = str(
                getattr(exc, "status_code", None)
                or getattr(exc, "code", None)
                or getattr(exc, "status", None)
                or ""
            ).upper()
            retryable = (
                isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                or "SERVERERROR" in type(exc).__name__.upper()
                or any(marker in error_message or marker in error_code for marker in (
                    "500", "INTERNAL", "503", "UNAVAILABLE", "429",
                    "RESOURCE_EXHAUSTED", "RATE_LIMIT", "TIMEOUT",
                    "DEADLINE_EXCEEDED"
                ))
            )

            logger.warning(
                "Gemini company extraction failed (%s) on attempt %s/%s%s.",
                type(exc).__name__,
                attempt + 1,
                max_attempts,
                "; retrying" if retryable and attempt < max_attempts - 1 else ""
            )
            _log_diagnostic(
                "gemini_extraction_failed", level="warning",
                error_type=type(exc).__name__, error_code=_safe_error_code(exc),
                attempt=attempt + 1, max_attempts=max_attempts,
                retrying=bool(retryable and attempt < max_attempts - 1)
            )

            if retryable and attempt < max_attempts - 1:
                await asyncio.sleep(min(0.5 * (2 ** attempt), 1.5))
                continue

            raise classify_external_error("Gemini", exc) from exc

    raise ExternalServiceError(
        "gemini_unavailable",
        "Gemini could not complete competitor discovery after multiple attempts.",
        503
    )


def find_official_website(
    company_name: str,
    search_results: list[dict]
) -> str | None:
    """
    Identify a likely official domain using distinctive
    company-name tokens in the hostname.

    This is a heuristic, not definitive verification.
    """

    company_key = normalize_company_name(company_name)
    if not company_key:
        return None
    compact_name = company_key.replace(" ", "")
    tokens = company_key.split()
    distinctive_tokens = [token for token in tokens if len(token) >= 4]
    scored_results = []
    for result in search_results:
        url = str(result.get("url", "")).strip()
        parsed = urlparse(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            continue
        domain = normalize_domain(url)

        if not domain or is_excluded_domain(domain):
            continue

        labels = domain.split(".")
        base_label = labels[-2] if len(labels) > 1 else labels[0]
        if len(labels) > 2 and labels[-2] in {"co", "com", "org", "net"}:
            base_label = labels[-3]
        base_key = normalize_company_name(base_label.replace("-", " "))
        domain_signal = (
            base_key == company_key
            or base_key.replace(" ", "") == compact_name
            or base_label in distinctive_tokens
        )
        if not domain_signal:
            continue
        normalized_title = normalize_company_name(result.get("title", ""))
        score = 0
        if base_label == compact_name:
            score += 5
        elif base_label in distinctive_tokens:
            score += 3
        if company_name_matches_text(company_name, result.get("title", "")):
            score += 3
        elif all(token in normalized_title.split() for token in tokens):
            score += 2

        if score >= 3:
            scored_results.append((score, root_url(url)))

    if not scored_results:
        return None
    scored_results.sort(key=lambda item: item[0], reverse=True)
    return scored_results[0][1]


def fallback_candidates_from_search_results(
    profile: dict,
    search_results: list[dict],
    limit: int = 8
) -> list[dict]:
    """Use supported result-site names as candidates when Gemini is unavailable.

    This deliberately produces only a candidate name and cited search URL.
    The normal company validation and fetched-page verification stages still
    determine whether any candidate can be returned as relevant evidence.
    """
    candidates = []
    seen_names = set()
    brand_name = str(profile.get("brand_name", ""))
    for result in search_results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        domain = normalize_domain(url)
        if not domain or is_excluded_domain(domain):
            continue

        labels = domain.split(".")
        base_label = labels[-2] if len(labels) > 1 else labels[0]
        if len(labels) > 2 and labels[-2] in {"co", "com", "org", "net"}:
            base_label = labels[-3]
        domain_name = base_label.replace("-", " ").strip()
        proposed_name = domain_name
        if not proposed_name:
            continue

        title = str(result.get("title", ""))
        content = str(result.get("content", ""))
        # Require the result text itself to identify the domain-derived name;
        # a URL alone is not enough to establish a candidate.
        if not company_name_matches_text(proposed_name, f"{title} {content}"):
            continue
        visible_name = re.search(re.escape(domain_name), f"{title} {content}", re.IGNORECASE)
        if visible_name:
            proposed_name = visible_name.group(0).strip()
        if normalize_company_name(proposed_name) == normalize_company_name(brand_name):
            continue

        name_key = normalize_company_name(proposed_name)
        if not name_key or name_key in seen_names:
            continue
        seen_names.add(name_key)
        candidates.append({
            "name": proposed_name,
            "product_name": "",
            "description": "",
            "competitor_type": "unclear",
            "competitor_type_reason": "Awaiting fetched-page evidence; this candidate was identified from a Tavily result.",
            "evidence_urls": [url]
        })
        if len(candidates) >= limit:
            break
    return candidates


async def discover_official_websites(
    companies: list[dict],
    api_key: str
) -> list[dict]:
    """Search for likely official websites for extracted companies."""

    async def search_company(company: dict) -> dict:
        name = str(company.get("name", "")).strip()

        if not name:
            company["official_website"] = None
            return company

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    search_tavily,
                    f"{name} official website",
                    api_key
                ),
                timeout=TAVILY_REQUEST_TIMEOUT
            )

            company["official_website"] = find_official_website(
                name,
                results
            )

        except Exception as exc:
            logger.warning("Official website lookup failed (%s).", type(exc).__name__)
            company["official_website"] = None

        return company

    return await asyncio.gather(
        *(search_company(company) for company in companies[:5])
    )
def _has_anchor_overlap(anchor: str, text: str) -> bool:
    """Require meaningful topic overlap instead of common-word coincidence."""
    stop_words = {
        "a", "an", "and", "for", "from", "in", "of", "on", "or", "the",
        "to", "with", "platform", "software", "solution", "solutions", "service"
    }
    def market_tokens(value: str) -> list[str]:
        normalized = str(value or "").casefold()
        normalized = re.sub(r"\be[\s-]?commerce\b", " commerce ", normalized)
        normalized = re.sub(r"\b(?:online|digital|electronic)\s+commerce\b", " commerce ", normalized)
        normalized = re.sub(r"\b(?:online|web)\s+stores?\b", " commerce ", normalized)
        normalized = re.sub(r"\bstorefronts?\b|\bshopping\s+carts?\b|\bcommerce\s+cloud\b", " commerce ", normalized)
        return re.findall(r"[a-z0-9]+", normalized)

    tokens = [
        token for token in market_tokens(anchor)
        if token not in stop_words
    ]
    text_tokens = set(market_tokens(text))
    overlap = set(tokens) & text_tokens
    # Commerce is a sufficiently specific market signal to connect terms
    # such as e-commerce, online store, storefront, and commerce cloud.
    if "commerce" in overlap:
        return True
    if len(tokens) >= 3:
        return len(overlap) >= 2
    if len(tokens) == 2:
        return len(overlap) == 2
    return bool(tokens and len(tokens[0]) >= 7 and tokens[0] in text_tokens)


def _same_sentence_mentions(
    sentence: str,
    company_name: str,
    product_name: str,
    brand_name: str
) -> bool:
    has_candidate = (
        company_name_matches_text(company_name, sentence)
        or bool(product_name and company_name_matches_text(product_name, sentence))
    )
    return has_candidate and bool(brand_name) and company_name_matches_text(brand_name, sentence)


def _relationship_marker_is_negated(sentence: str, marker: str) -> bool:
    """Avoid treating an explicit denial as a competitor relationship."""
    index = sentence.casefold().find(marker.casefold())
    if index < 0:
        return False
    prefix = sentence[:index]
    return bool(re.search(
        r"\b(?:not|never|no longer|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b[^.!?]{0,40}$",
        prefix,
        flags=re.IGNORECASE
    ))


def classify_competitor_type_from_sources(
    company_name: str,
    product_name: str,
    brand_name: str,
    sources: list[dict]
) -> tuple[str, str]:
    """Use source wording for direct/indirect labels; otherwise remain unclear."""
    direct_markers = (
        "direct competitor", "direct alternative", "competes with",
        "compete with", "competing with",
        "competes against", "compete against", "competing against"
    )
    indirect_markers = (
        "indirect competitor", "indirect alternative", "indirect substitute"
    )
    for source in sources:
        text = f"{source.get('title', '')}. {source.get('content', '')}"
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            normalized_sentence = f" {sentence.casefold()} "
            if not _same_sentence_mentions(
                sentence, company_name, product_name, brand_name
            ):
                continue
            source_url = str(source.get("url", ""))
            if any(
                marker in normalized_sentence
                and not _relationship_marker_is_negated(sentence, marker)
                for marker in indirect_markers
            ):
                marker = next(
                    marker.strip()
                    for marker in indirect_markers
                    if marker in normalized_sentence
                    and not _relationship_marker_is_negated(sentence, marker)
                )
                return (
                    "indirect",
                    f"The fetched page describes {company_name} and {brand_name} using “{marker}” ({source_url})."
                )
            if any(
                marker in normalized_sentence
                and not _relationship_marker_is_negated(sentence, marker)
                for marker in direct_markers
            ):
                marker = next(
                    marker.strip()
                    for marker in direct_markers
                    if marker in normalized_sentence
                    and not _relationship_marker_is_negated(sentence, marker)
                )
                return (
                    "direct",
                    f"The fetched page links {company_name} with {brand_name} using “{marker}” ({source_url}); the direct label follows that wording."
                )
    return (
        "unclear",
        f"Sources identify {company_name} and a related offering, but do not establish whether it competes directly or indirectly with {brand_name or 'the analyzed brand'}."
    )


def build_evidence_description(
    company_name: str,
    product_name: str,
    source_texts: list[str],
    relevance_anchors: list[str]
) -> str:
    """Choose a source sentence that supports the candidate's offering."""
    sentences = [
        sentence.strip()
        for source_text in source_texts
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text)
        if sentence.strip()
    ]
    supported = [
        sentence for sentence in sentences
        if (
            company_name_matches_text(company_name, sentence)
            or bool(product_name and company_name_matches_text(product_name, sentence))
        ) and any(
            company_name_matches_text(anchor, sentence)
            or _has_anchor_overlap(anchor, sentence)
            for anchor in relevance_anchors
        )
    ]
    supported = list(dict.fromkeys(supported))
    if supported:
        return " ".join(supported)[:500]
    return ""


def deduplicate_companies(companies: list[dict]) -> list[dict]:
    """Deduplicate extracted candidates by normalized name and official domain."""
    unique = []
    indexes = {}
    for company in companies:
        if not isinstance(company, dict):
            continue
        name = str(company.get("name", "")).strip()
        name_key = normalize_company_name(name)
        domain_key = normalize_domain(str(company.get("official_website") or ""))
        if not name_key:
            continue
        key = name_key
        if key not in indexes and domain_key:
            existing_by_domain = next((
                index for index, item in enumerate(unique)
                if normalize_domain(str(item.get("official_website") or "")) == domain_key
            ), None)
            if existing_by_domain is not None:
                key = normalize_company_name(unique[existing_by_domain]["name"])
        if key not in indexes:
            indexes[key] = len(unique)
            item = dict(company)
            item["name"] = name
            item["normalized_name"] = name_key
            unique.append(item)
            continue
        existing = unique[indexes[key]]
        for field in ("product_name", "description", "official_website"):
            if not existing.get(field) and company.get(field):
                existing[field] = company[field]
        existing_urls = existing.get("evidence_urls", [])
        company_urls = company.get("evidence_urls", [])
        if not isinstance(existing_urls, list):
            existing_urls = []
        if not isinstance(company_urls, list):
            company_urls = []
        existing["evidence_urls"] = list(dict.fromkeys(
            existing_urls + company_urls
        ))
    return unique


def official_website_is_validated(
    official_website: str | None,
    evidence_details: list[dict]
) -> bool:
    """Accept a proposed official site only when its page identifies the company."""
    domain = normalize_domain(str(official_website or ""))
    if not domain:
        return False
    return any(
        detail.get("accessible")
        and detail.get("classification") in {
            "company_mentioned",
            "company_and_product_mentioned",
            "product_or_capability_comparison_found",
            "explicit_competitor_relationship_found",
        }
        and normalize_domain(str(detail.get("source_url") or detail.get("url") or "")) == domain
        for detail in evidence_details
    )


def validate_company_evidence(
    company: dict,
    search_results: list[dict],
    profile: dict | None = None
) -> dict:
    """
    Evaluate whether cited search snippets support a
    company's relevance to the analyzed brand.

    Scores are heuristic evidence indicators, not proof.
    """

    name = str(company.get("name", "")).strip()

    if not name:
        return {}

    result_lookup = {}

    for result in search_results:
        canonical = canonicalize_url(result.get("url", ""))

        if canonical:
            result_lookup[canonical] = result

    cited_urls = company.get("evidence_urls", [])

    if not isinstance(cited_urls, list):
        cited_urls = []

    cited_results = []
    valid_evidence = []

    for url in cited_urls:
        canonical = canonicalize_url(str(url))

        if canonical in result_lookup and canonical not in valid_evidence:
            valid_evidence.append(canonical)
            cited_results.append(result_lookup[canonical])

    if not cited_results:
        return {
            "evidence_urls": [],
            "evidence_confidence": 0.0,
            "evidence_status": "unsupported",
            "competitor_type": "unclear",
            "product_name": "",
            "description": "",
            "competitor_type_reason": "No cited search result could be matched to this candidate."
        }

    # Build terms associated with this company.
    product_name = str(company.get("product_name", "")).strip()
    profile = profile or {}
    company_results = [
        result for result in cited_results
        if company_name_matches_text(
            name,
            f"{result.get('title', '')} {result.get('content', '')} "
            f"{normalize_domain(result.get('url', '')).replace('-', ' ')}"
        )
    ]
    company_mentioned = bool(company_results)
    if not company_mentioned:
        return {
            "evidence_urls": [],
            "evidence_confidence": 0.0,
            "evidence_status": "unsupported",
            "competitor_type": "unclear",
            "product_name": "",
            "description": "",
            "competitor_type_reason": "The cited search text does not identify this company."
        }

    cited_results = company_results
    valid_evidence = [
        canonicalize_url(result.get("url", ""))
        for result in cited_results
        if canonicalize_url(result.get("url", ""))
    ]
    context_texts = [
        f"{result.get('title', '')}. {result.get('content', '')}"
        for result in cited_results
    ]
    context_text = "\n".join(context_texts)
    context_sentences = [
        sentence.strip()
        for context in context_texts
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", context)
        if sentence.strip()
    ]
    product_supported = bool(product_name) and any(
        company_name_matches_text(name, sentence)
        and company_name_matches_text(product_name, sentence)
        for sentence in context_sentences
    )
    if not product_supported:
        product_name = ""

    brand_offerings = profile.get("products_services", [])
    if not isinstance(brand_offerings, list):
        brand_offerings = [str(brand_offerings)] if brand_offerings else []
    category = str(profile.get("category", "")).strip()
    relevance_anchors = [str(item).strip() for item in brand_offerings if str(item).strip()]
    if category:
        relevance_anchors.append(category)

    candidate_sentences = [
        sentence for sentence in context_sentences
        if company_name_matches_text(name, sentence)
    ]
    brand_name = str(profile.get("brand_name", ""))
    comparison_markers = (
        "versus", "vs", "compared to", "compared with",
        "direct competitor", "indirect competitor", "competitor of",
        "competitor to", "competes with", "compete with", "competing with",
        "competes against", "compete against", "competing against",
        "direct alternative", "indirect alternative", "indirect substitute",
        "alternative to", "alternatives to"
    )
    comparison_sentences = [
        sentence for sentence in candidate_sentences
        if _same_sentence_mentions(sentence, name, product_name, brand_name)
        and any(
            re.search(r"(?<![a-z0-9])" + re.escape(marker) + r"(?![a-z0-9])", sentence, re.IGNORECASE)
            and not _relationship_marker_is_negated(sentence, marker)
            for marker in comparison_markers
        )
    ]
    comparison_supported = bool(comparison_sentences)
    relevant_anchors = [
        anchor for anchor in relevance_anchors
        if any(
            company_name_matches_text(anchor, sentence)
            or _has_anchor_overlap(anchor, sentence)
            for sentence in candidate_sentences
        )
    ]
    if not relevant_anchors and not comparison_supported:
        return {
            "evidence_urls": [],
            "evidence_confidence": 0.0,
            "evidence_status": "unrelated",
            "competitor_type": "unclear",
            "product_name": product_name,
            "description": "",
            "competitor_type_reason": "The cited text identifies the company but does not connect its offering to the analyzed business."
        }

    relationship_type, relationship_reason = classify_competitor_type_from_sources(
        name,
        product_name,
        brand_name,
        cited_results
    )
    if product_supported:
        status = "company_and_product_supported"
        confidence = 0.75
    else:
        status = "company_and_relevant_market_supported"
        confidence = 0.55
    description = (
        build_evidence_description(name, product_name, context_texts, relevant_anchors)
        if relevant_anchors
        else " ".join(comparison_sentences[:1])[:500]
    )

    return {
        "evidence_urls": valid_evidence,
        "evidence_confidence": confidence,
        "evidence_status": status,
        "competitor_type": relationship_type,
        "product_name": product_name,
        "description": description,
        "competitor_type_reason": relationship_reason,
        "relevance_reason": (
            f"Cited search text explicitly compares {name} with {brand_name} at {valid_evidence[0]}; retain for evidence review."
            if comparison_supported and not relevant_anchors
            else (
                f"Cited search text identifies {name}"
                + (f" and the product {product_name}" if product_name else "")
                + "; its text also overlaps with "
                + ", ".join(relevant_anchors[:2]) + "."
            )
        )
    }


def rank_candidates_by_evidence(candidates: list[dict]) -> list[dict]:
    """Keep candidates with qualifying fetched evidence; use semantic score only as a tie-breaker."""
    evidence_tiers = {
        "explicit_competitor_relationship_found": 4,
        "product_or_capability_comparison_found": 3,
        "company_and_product_mentioned": 2,
        "company_mentioned": 1,
    }
    eligible = []
    for candidate in candidates:
        try:
            accessible_count = int(candidate.get("evidence_pages_accessible", 0) or 0)
            evidence_score = float(candidate.get(
                "heuristic_evidence_score",
                candidate.get("evidence_confidence", 0.0)
            ) or 0.0)
        except (TypeError, ValueError):
            continue
        if accessible_count <= 0 or evidence_score <= 0:
            continue
        if candidate.get("evidence_status") not in evidence_tiers:
            continue

        details = candidate.get("evidence_details", [])
        if not isinstance(details, list):
            continue
        has_fetched_evidence = any(
            detail.get("accessible") is True
            and detail.get("classification") in evidence_tiers
            and any(
                isinstance(excerpt, dict)
                and isinstance(excerpt.get("matched_text"), str)
                and excerpt["matched_text"].strip()
                and isinstance(excerpt.get("source_url"), str)
                and excerpt["source_url"].strip()
                for excerpt in detail.get("excerpts", [])
            )
            for detail in details
            if isinstance(detail, dict)
        )
        if not has_fetched_evidence:
            continue
        eligible.append(candidate)

    def rank_key(candidate: dict) -> tuple:
        details = candidate.get("evidence_details", [])
        if not isinstance(details, list):
            details = []
        excerpts = [
            excerpt
            for detail in details if isinstance(detail, dict)
            for excerpt in detail.get("excerpts", [])
            if isinstance(excerpt, dict)
        ]
        relationship_count = sum(
            1 for excerpt in excerpts
            if excerpt.get("classification") == "explicit_competitor_relationship_found"
        )
        try:
            similarity_score = float(candidate.get("similarity_score", 0.0))
        except (TypeError, ValueError):
            similarity_score = 0.0
        return (
            evidence_tiers.get(str(candidate.get("evidence_status", "")), 0),
            int(candidate.get("evidence_pages_accessible", 0) or 0),
            relationship_count,
            similarity_score
        )

    return sorted(eligible, key=rank_key, reverse=True)


async def discover_competitors(
    analysis: dict,
    website_url: str = ""
) -> list[dict]:
    """
    Discover potential competitors using Tavily,
    extract companies using Gemini, validate evidence,
    and rank using Sentence Transformers.
    """

    api_key = require_api_key("TAVILY_API_KEY")

    profile = analysis.get("competitor_analysis", {})

    if not profile:
        logger.warning("Competitor analysis is missing.")
        _log_discovery_summary(0, reason="missing_profile")
        return []

    brand_name = str(profile.get("brand_name", ""))
    brand_domain = normalize_domain(website_url)

    queries = build_search_queries(profile)

    if not queries:
        _log_discovery_summary(0, reason="no_search_queries")
        return []

    # 1. Search for potential competitors with per-query deadlines.
    async def search_with_timeout(query: str, query_index: int) -> list[dict]:
        _log_diagnostic("tavily_search_started", query_index=query_index)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(search_tavily, query, api_key),
                timeout=TAVILY_REQUEST_TIMEOUT
            )
            _log_diagnostic(
                "tavily_search_completed", query_index=query_index,
                search_result_count=len(results)
            )
            return results
        except asyncio.TimeoutError as error:
            _log_diagnostic(
                "tavily_search_failed", level="warning", query_index=query_index,
                error_type=type(error).__name__, error_code="tavily_timeout"
            )
            raise classify_external_error("Tavily", error) from error
        except Exception as error:
            _log_diagnostic(
                "tavily_search_failed", level="warning", query_index=query_index,
                error_type=type(error).__name__, error_code=_safe_error_code(error)
            )
            if isinstance(error, ExternalServiceError):
                raise
            raise classify_external_error("Tavily", error) from error

    _log_diagnostic("tavily_search_batch_started", query_count=len(queries))
    search_tasks = [
        search_with_timeout(query, index) for index, query in enumerate(queries, 1)
    ]

    responses = await asyncio.gather(
        *search_tasks,
        return_exceptions=True
    )

    search_results = []
    search_failures = []

    for response in responses:
        if isinstance(response, Exception):
            logger.warning("Tavily search failed (%s).", type(response).__name__)
            search_failures.append(response)
            continue

        search_results.extend(response)

    _log_diagnostic(
        "tavily_search_batch_completed", search_result_count=len(search_results),
        search_failure_count=len(search_failures)
    )

    if not search_results:
        if search_failures:
            first_error = search_failures[0]
            if isinstance(first_error, ExternalServiceError):
                raise first_error
            raise classify_external_error("Tavily", first_error)
        _log_discovery_summary(0, reason="no_search_results")
        return []

    if search_failures and isinstance(analysis, dict):
        analysis["competitor_discovery_status"] = {
            "status": "partial",
            "message": "Some Tavily searches failed; results may be incomplete."
        }

    # 2. Remove duplicate domains and excluded websites.
    unique_results = []
    seen_result_urls = set()
    filtered_search_results = 0

    for result in search_results:
        url = result.get("url", "")
        domain = normalize_domain(url)

        if not domain:
            filtered_search_results += 1
            continue

        if domain == brand_domain:
            filtered_search_results += 1
            continue

        if is_excluded_domain(domain):
            filtered_search_results += 1
            continue

        canonical = canonicalize_url(url)
        if not canonical or canonical in seen_result_urls:
            filtered_search_results += 1
            continue

        seen_result_urls.add(canonical)
        unique_results.append(result)

    _log_diagnostic(
        "search_results_filtered", input_result_count=len(search_results),
        candidate_source_count=len(unique_results),
        removed_result_count=filtered_search_results
    )

    if not unique_results:
        _log_discovery_summary(0, reason="no_unique_search_results")
        return []

    # 3. Extract company names using Gemini.
    gemini_fallback_used = False
    try:
        companies = await extract_companies_with_gemini(profile, unique_results)
    except ExternalServiceError as error:
        gemini_fallback_used = True
        companies = fallback_candidates_from_search_results(profile, unique_results)
        _log_diagnostic(
            "gemini_fallback_candidates_created",
            error_type=type(error).__name__, error_code=error.code,
            candidate_count=len(companies)
        )
        analysis["competitor_discovery_status"] = {
            "status": "partial",
            "message": "Gemini extraction was unavailable; Tavily candidates are being verified."
        }

    if not companies:
        _log_diagnostic("company_candidates_extracted", candidate_count=0)
        _log_discovery_summary(
            0,
            reason=("gemini_failed_no_valid_tavily_candidates" if gemini_fallback_used else "no_company_candidates")
        )
        return []

    extracted_count = len(companies)
    _log_diagnostic("company_candidates_extracted", candidate_count=extracted_count)
    companies = deduplicate_companies(companies)
    _log_diagnostic(
        "company_candidates_deduplicated", input_candidate_count=extracted_count,
        candidate_count=len(companies), removed_candidate_count=extracted_count - len(companies)
    )

    # 4. Search for likely official company websites.
    companies = await discover_official_websites(
        companies,
        api_key
    )
    _log_diagnostic(
        "official_websites_identified", candidate_count=len(companies),
        official_website_count=sum(bool(item.get("official_website")) for item in companies)
    )

    # 5. Validate evidence against actual Tavily results.
    validated_companies = []
    removed_by_validation = 0

    for company in companies:
        validation = validate_company_evidence(
            company,
            unique_results,
            profile=profile
        )

        if not validation:
            removed_by_validation += 1
            continue

        if validation["evidence_status"] in {"unsupported", "unrelated"}:
            removed_by_validation += 1
            continue

        company["ai_competitor_type"] = str(
            company.get("competitor_type", "unclear")
        ).lower()
        company.update(validation)
        validated_companies.append(company)

    _log_diagnostic(
        "search_evidence_candidates_validated", input_candidate_count=len(companies),
        candidate_count=len(validated_companies),
        removed_candidate_count=removed_by_validation
    )

    validated_candidate_count = len(validated_companies)
    companies = deduplicate_companies(validated_companies)
    _log_diagnostic(
        "validated_candidates_deduplicated",
        input_candidate_count=validated_candidate_count,
        candidate_count=len(companies),
        removed_candidate_count=validated_candidate_count - len(companies)
    )

    # 6. Remove invalid and duplicate candidates.
    final_candidates = []
    seen_companies = set()
    removed_final_candidates = 0

    for company in companies:
        name = str(company.get("name", "")).strip()
        website = company.get("official_website")

        if not name:
            removed_final_candidates += 1
            continue

        name_key = normalize_company_name(name)
        if name_key == normalize_company_name(brand_name):
            removed_final_candidates += 1
            continue

        if website and normalize_domain(website) == brand_domain:
            removed_final_candidates += 1
            continue

        if name_key in seen_companies:
            removed_final_candidates += 1
            continue

        seen_companies.add(name_key)
        evidence_urls = company.get("evidence_urls", [])

        if not isinstance(evidence_urls, list):
            evidence_urls = []

        # Include the discovered official website for page verification.
        if website:
            evidence_urls.append(website)

        # Remove duplicate URLs while preserving order.
        unique_evidence_urls = []
        seen_evidence_urls = set()
        for url in evidence_urls:
            if not isinstance(url, str) or not url.strip():
                continue
            canonical = canonicalize_url(url)
            if canonical and canonical not in seen_evidence_urls:
                seen_evidence_urls.add(canonical)
                unique_evidence_urls.append(canonical)
        evidence_urls = unique_evidence_urls
        final_candidates.append({
            "name": name,
            "normalized_name": name_key,
            "url": website,
            "official_website": website,
            "product_name": company.get("product_name", ""),
            "description": str(
                company.get("description", "")
            ),
            "relevance_reason": company.get("relevance_reason", ""),
            "competitor_type": "unclear",
            "ai_competitor_type": company.get("ai_competitor_type", "unclear"),
            "competitor_type_reason": "Awaiting accessible webpage evidence for a direct or indirect classification.",
            "evidence_urls": evidence_urls,
            "description_source_urls": company.get("evidence_urls", []),
            "evidence_confidence": company.get(
                "evidence_confidence",
                0.0
            ),
            "evidence_status": company.get(
                "evidence_status",
                "unsupported"
            ),
            "source": "Tavily search + Gemini extraction",
            "verification_status": (
                "official_website_candidate"
                if website
                else "website_not_identified"
            )
        })

    _log_diagnostic(
        "final_candidate_filtering_completed", input_candidate_count=len(companies),
        candidate_count=len(final_candidates), removed_candidate_count=removed_final_candidates
    )

    # Stop if no valid candidates were discovered.
    if not final_candidates:
        _log_discovery_summary(0, reason="no_valid_candidates")
        return []

    # 7. Verify competitor evidence using actual webpage content.
    relevance_anchors = profile.get("products_services", [])
    if not isinstance(relevance_anchors, list):
        relevance_anchors = [str(relevance_anchors)] if relevance_anchors else []
    category = str(profile.get("category", "")).strip()
    if category:
        relevance_anchors = [*relevance_anchors, category]
    final_candidates = await verify_competitors(
        final_candidates,
        analyzed_brand_name=brand_name,
        relevance_anchors=[
            str(anchor).strip()
            for anchor in relevance_anchors
            if str(anchor).strip()
        ]
    )
    evidence_supported_count = sum(
        int(candidate.get("evidence_pages_accessible", 0) or 0) > 0
        and candidate.get("evidence_status") not in {
            "no_relevant_evidence", "inaccessible", "no_usable_evidence", "verification_error"
        }
        for candidate in final_candidates
    )
    _log_diagnostic(
        "evidence_verification_completed", input_candidate_count=len(final_candidates),
        evidence_supported_candidate_count=evidence_supported_count
    )

    # Validate official-domain candidates against fetched page evidence, and
    # only assign direct/indirect when an actual page makes that relation.
    official_site_candidates = 0
    official_sites_validated = 0
    for candidate in final_candidates:
        official_website = candidate.get("official_website")
        official_domain = normalize_domain(str(official_website or ""))
        details = candidate.get("evidence_details", [])
        validated_official_domain = ""
        if official_website:
            official_site_candidates += 1
            if official_website_is_validated(official_website, details):
                official_sites_validated += 1
                candidate["verification_status"] = "official_website_validated"
                validated_official_domain = official_domain
            else:
                candidate["official_website"] = None
                candidate["url"] = None
                candidate["verification_status"] = "official_website_unverified"

        for detail in details:
            detail_domain = normalize_domain(str(detail.get("source_url") or detail.get("url") or ""))
            is_validated_official = bool(
                validated_official_domain
                and detail_domain == validated_official_domain
            )
            is_unverified_official_candidate = bool(
                official_domain and detail_domain == official_domain
            )
            detail["is_official_domain"] = is_validated_official
            detail["source_type"] = (
                "official_website" if is_validated_official
                else "unverified_source" if is_unverified_official_candidate
                else "third_party" if official_domain
                else "unverified_source"
            )
            for excerpt in detail.get("excerpts", []):
                excerpt["source_type"] = detail["source_type"]

        relationship_sources = [
            {
                "url": excerpt.get("source_url", ""),
                "content": excerpt.get("matched_text", "")
            }
            for detail in details
            for excerpt in detail.get("excerpts", [])
            if excerpt.get("classification") == "explicit_competitor_relationship_found"
        ]
        competitor_type, type_reason = classify_competitor_type_from_sources(
            candidate.get("name", ""),
            candidate.get("product_name", ""),
            brand_name,
            relationship_sources
        )
        candidate["competitor_type"] = competitor_type
        candidate["competitor_type_reason"] = type_reason
        if candidate.get("description"):
            candidate["description_is_source_derived"] = True

    _log_diagnostic(
        "official_website_validation_completed",
        official_website_candidate_count=official_site_candidates,
        validated_official_website_count=official_sites_validated,
        unvalidated_official_website_count=official_site_candidates - official_sites_validated
    )

    # Only candidates with relevant excerpts from successfully fetched pages
    # can appear in the competitor results. Search snippets and inaccessible
    # pages remain discovery inputs, not verified competitor evidence.
    final_candidates = rank_candidates_by_evidence(final_candidates)
    if not final_candidates:
        _log_discovery_summary(0, reason="no_accessible_relevant_evidence")
        return []
    final_candidates = deduplicate_competitor_excerpts(final_candidates)

    # 8. Rank company descriptions using Sentence Transformers.
    try:
        from app.services.semantic_matcher import rank_similar_items

        products = profile.get("products_services", [])

        if isinstance(products, list):
            products_text = " ".join(
                str(item) for item in products
            )
        else:
            products_text = str(products)

        brand_profile = " ".join([
            brand_name,
            str(profile.get("category", "")),
            str(profile.get("overview", "")),
            products_text
        ])

        descriptions = [
            f"{item['name']}. {item['description']}"
            for item in final_candidates
        ]

        ranked_results = await asyncio.to_thread(
            rank_similar_items,
            brand_profile,
            descriptions
        )

        score_lookup = {
            item["description"]: item["similarity_score"]
            for item in ranked_results
        }

        for candidate, description in zip(
            final_candidates,
            descriptions
        ):
            candidate["similarity_score"] = score_lookup.get(
                description,
                0.0
            )

        discovered = final_candidates[:5]
        _log_discovery_summary(len(discovered))
        return discovered
    except Exception as exc:
        logger.warning("Semantic ranking failed (%s).", type(exc).__name__)
        discovered = rank_candidates_by_evidence(final_candidates)[:5]
        _log_discovery_summary(
            len(discovered), reason="semantic_ranking_failed_using_evidence_order",
            error_code=_safe_error_code(exc)
        )
        return discovered
