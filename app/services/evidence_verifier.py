

import asyncio
import ipaddress
import logging
import socket
import re
from difflib import SequenceMatcher

from urllib.parse import urlsplit, urljoin

import httpx
from bs4 import BeautifulSoup

from app.utils.company_names import company_name_matches_text, normalize_company_name


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _evidence_url_key(url: str) -> str:
    """Canonical key for deduplicating equivalent evidence URLs."""
    try:
        parsed = urlsplit(str(url).strip())
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return ""
        host = parsed.hostname.lower().removeprefix("www.")
        port = parsed.port
        if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        return f"{parsed.scheme.lower()}://{host}{parsed.path.rstrip('/')}"
    except (TypeError, ValueError):
        return ""

MAX_PAGES = 3
MAX_PAGE_SIZE = 1_000_000
REQUEST_TIMEOUT = 10
MAX_REDIRECTS = 5
MAX_CONCURRENT_CANDIDATES = 3

COMPETITOR_TERMS = [
    "direct competitor",
    "indirect competitor",
    "direct alternative",
    "indirect alternative",
    "competitor of",
    "competitors of",
    "competitor to",
    "competes with",
    "compete with",
    "competing with",
    "competes against",
    "compete against",
    "competing against",
]

COMPARISON_TERMS = [
    "compared to",
    "compared with",
    "compared against",
    "comparison",
    "compare",
    "versus",
    "vs",
    "similar to",
    "comparable to",
    "alternatives to",
    "alternative to",
]

TRUSTED_REPORT_DOMAINS = {
    "gartner.com",
    "forrester.com",
    "idc.com",
    "statista.com",
    "digitalcommerce360.com",
}


def _market_tokens(value: str) -> set[str]:
    normalized = str(value or "").casefold()
    normalized = re.sub(r"\be[\s-]?commerce\b", " commerce ", normalized)
    normalized = re.sub(r"\b(?:online|digital|electronic)\s+commerce\b", " commerce ", normalized)
    normalized = re.sub(r"\b(?:online|web)\s+stores?\b", " commerce ", normalized)
    normalized = re.sub(r"\bstorefronts?\b|\bshopping\s+carts?\b|\bcommerce\s+cloud\b", " commerce ", normalized)
    return set(re.findall(r"[a-z0-9]+", normalized))


def _market_relevance_matches(anchor: str, sentence: str) -> bool:
    """Check exact topic overlap, with common commerce wording normalized."""
    anchor_tokens = _market_tokens(anchor)
    sentence_tokens = _market_tokens(sentence)
    if not anchor_tokens or not sentence_tokens:
        return False
    if "commerce" in anchor_tokens:
        commerce_context = {
            "platform", "store", "storefront", "merchant", "merchants",
            "retail", "retailer", "retailers", "shopping", "cart", "carts",
            "checkout", "catalog", "payment", "payments", "b2b", "b2c",
            "omnichannel", "multichannel"
        }
        return (
            "commerce" in sentence_tokens
            and bool(commerce_context & sentence_tokens)
        )
    meaningful = anchor_tokens - {
        "a", "an", "and", "for", "from", "in", "of", "on", "or", "the",
        "to", "with", "platform", "software", "solution", "solutions",
        "service", "services", "tool", "tools", "product", "products"
    }
    overlap = meaningful & sentence_tokens
    return len(overlap) >= (2 if len(meaningful) >= 3 else 1)


def _excerpt_fingerprint(text: str) -> tuple[str, tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", str(text).casefold())
    return " ".join(tokens), tuple(token for token in tokens if token.isdigit())


def _semantic_excerpt_tokens(text: str) -> set[str]:
    normalized = str(text).casefold()
    normalized = re.sub(r"\be[\s-]?commerce\b", " commerce ", normalized)
    normalized = re.sub(r"\b(?:online|digital)\s+commerce\b", " commerce ", normalized)
    normalized = re.sub(r"\b(?:online|web)\s+stores?\b", " commerce ", normalized)
    aliases = {
        "offers": "provide",
        "provides": "provide",
        "providing": "provide",
        "gives": "provide",
        "delivers": "provide",
        "supplies": "provide",
    }
    ignored = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "has",
        "in", "is", "it", "of", "on", "or", "the", "this", "to", "with",
        "according", "also", "that", "their", "these", "those"
    }
    return {
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in ignored and not token.isdigit()
    }


def _market_share_finding_signature(text: str) -> tuple[set[str], set[str]] | None:
    """Capture subjects and percentages for paraphrased relative-share claims."""
    normalized = str(text or "")
    lowered = normalized.casefold()
    if not re.search(r"\b(?:share|market|lead|ahead|followed)\b|%", lowered):
        return None
    # Combined market-control statements make a distinct aggregate claim and
    # must not be collapsed into an individual vendor-share comparison.
    if re.search(r"\b(?:together|combined|collectively|jointly)\b", lowered):
        return None
    percentages = set(re.findall(r"\b\d+(?:\.\d+)?\s*%", lowered))
    if not percentages:
        return None
    subjects = {
        token.casefold()
        for token in re.findall(r"\b[A-Z][A-Za-z0-9&'-]*\b", normalized)
        if token.casefold() not in {
            "us", "uk", "the", "with", "in", "of", "at", "for", "and",
            "according", "year", "stores"
        }
    }
    if len(subjects) < 2:
        return None
    return subjects, percentages


def _same_market_share_finding(left_text: str, right_text: str) -> bool:
    left = _market_share_finding_signature(left_text)
    right = _market_share_finding_signature(right_text)
    if not left or not right:
        return False
    left_subjects, left_percentages = left
    right_subjects, right_percentages = right
    return (
        len(left_subjects & right_subjects) >= 2
        and bool(left_percentages & right_percentages)
    )


def _specific_product_name(company_name: str, product_name: str) -> bool:
    company_tokens = set(re.findall(r"[a-z0-9]+", company_name.casefold()))
    product_tokens = set(re.findall(r"[a-z0-9]+", product_name.casefold()))
    product_tokens -= company_tokens
    product_tokens -= {
        "app", "apps", "cloud", "platform", "plus", "pro", "product",
        "service", "services", "software", "suite", "tool", "tools"
    }
    return bool(product_tokens)


def _is_duplicate_excerpt(text: str, existing_texts: list[str]) -> bool:
    fingerprint, numbers = _excerpt_fingerprint(text)
    if not fingerprint:
        return False
    tokens = fingerprint.split()
    for existing in existing_texts:
        previous, previous_numbers = _excerpt_fingerprint(existing)
        if fingerprint == previous:
            return True
        if _same_market_share_finding(text, existing):
            return True
        previous_tokens = previous.split()
        if numbers == previous_numbers and min(len(tokens), len(previous_tokens)) >= 7:
            if SequenceMatcher(None, fingerprint, previous).ratio() >= 0.92:
                return True
            duplicate_stopwords = {
                "a", "an", "and", "as", "at", "by", "for", "from", "in",
                "is", "of", "on", "or", "the", "to", "with", "according"
            }
            left = set(tokens) - duplicate_stopwords
            right = set(previous_tokens) - duplicate_stopwords
            if left and right and len(left & right) / len(left | right) >= 0.88:
                return True
            semantic_left = _semantic_excerpt_tokens(text)
            semantic_right = _semantic_excerpt_tokens(existing)
            if (
                numbers == previous_numbers
                and min(len(semantic_left), len(semantic_right)) >= 8
                and semantic_left
                and semantic_right
                and len(semantic_left & semantic_right) / len(semantic_left | semantic_right) >= 0.84
            ):
                return True
    return False


def deduplicate_competitor_excerpts(competitors: list[dict]) -> list[dict]:
    """Avoid repeating the same observed statement across competitor cards."""
    seen_excerpt_texts = []
    for competitor in competitors:
        for detail in competitor.get("evidence_details", []):
            kept_excerpts = []
            omitted = int(detail.get("duplicate_excerpts_omitted", 0) or 0)
            for excerpt in detail.get("excerpts", []):
                matched_text = excerpt.get("matched_text", "")
                if _is_duplicate_excerpt(matched_text, seen_excerpt_texts):
                    omitted += 1
                    continue
                kept_excerpts.append(excerpt)
                seen_excerpt_texts.append(matched_text)
            detail["excerpts"] = kept_excerpts
            if omitted:
                detail["duplicate_excerpts_omitted"] = omitted
    return competitors


def _source_priority(url: str) -> tuple[int, int]:
    """Prefer known industry reports and comparison/review pages by URL signal."""
    try:
        parsed = urlsplit(url)
        domain = (parsed.hostname or "").lower().removeprefix("www.")
        trusted = any(
            domain == item or domain.endswith("." + item)
            for item in TRUSTED_REPORT_DOMAINS
        )
        path = parsed.path.casefold()
        comparison = any(
            marker in path
            for marker in ("compar", "-vs-", "_vs_", "versus", "alternative", "review")
        )
        report = any(
            marker in path
            for marker in ("report", "research", "market", "share", "industry")
        )
        if trusted:
            return (0, 0)
        if comparison:
            return (1, 0)
        if report:
            return (2, 0)
        return (3, 0)
    except (TypeError, ValueError):
        return (4, 0)


def _positive_competitor_terms(sentence: str) -> list[str]:
    """Return explicit relationship phrases that are not negated in context."""
    positive_terms = []
    negation_pattern = re.compile(
        r"\b(?:not|never|no longer|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b[^.!?]{0,40}$",
        re.IGNORECASE
    )
    for term in COMPETITOR_TERMS:
        for match in re.finditer(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
            sentence,
            flags=re.IGNORECASE
        ):
            if not negation_pattern.search(sentence[:match.start()]):
                positive_terms.append(term)
                break
    return positive_terms


def _mentioned_competitor_terms(sentence: str) -> list[str]:
    """Find relationship wording even when the sentence explicitly denies it."""
    return [
        term for term in COMPETITOR_TERMS
        if re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
            sentence,
            flags=re.IGNORECASE
        )
    ]


def _positive_comparison_terms(sentence: str) -> list[str]:
    """Return comparative wording only when it connects both named subjects."""
    positive_terms = []
    negation_pattern = re.compile(
        r"\b(?:not|never|no longer|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b[^.!?]{0,40}$",
        re.IGNORECASE
    )
    for term in COMPARISON_TERMS:
        for match in re.finditer(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])",
            sentence,
            flags=re.IGNORECASE
        ):
            if not negation_pattern.search(sentence[:match.start()]):
                positive_terms.append(term)
                break
    return positive_terms


# --------------------------------------------------
# 1. PUBLIC URL VALIDATION
# --------------------------------------------------

def is_public_url(url: str) -> bool:
    """
    Allow only HTTP/HTTPS URLs resolving to public IP addresses.
    Reject private, local and reserved network addresses.
    """

    try:
        parsed = urlsplit(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not parsed.hostname:
            return False

        if parsed.username or parsed.password:
            return False

        hostname = parsed.hostname.lower()

        if hostname == "localhost" or hostname.endswith(".local"):
            return False

        try:
            ip = ipaddress.ip_address(hostname)

            if not ip.is_global:
                return False

        except ValueError:
            addresses = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM
            )

            if not addresses:
                return False

            for address in addresses:
                resolved_ip = ipaddress.ip_address(address[4][0])

                if not resolved_ip.is_global:
                    return False

        return True

    except Exception:
        return False


# --------------------------------------------------
# 2. HTML TEXT EXTRACTION
# --------------------------------------------------

def extract_page_text(html: str) -> str:
    """Extract readable text from an HTML page."""

    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav",
        "svg",
        "form"
    ]):
        element.decompose()

    blocks = [
        element.get_text(" ", strip=True)
        for element in soup.find_all([
            "h1", "h2", "h3", "h4", "h5", "h6",
            "p", "li", "blockquote", "td", "th"
        ])
        if element.get_text(" ", strip=True)
    ]
    if blocks:
        return "\n".join(blocks)
    return " ".join(soup.stripped_strings)


# --------------------------------------------------
# 3. FETCH EVIDENCE PAGE WITH SAFE REDIRECTS
# --------------------------------------------------

async def fetch_evidence_page(
    client: httpx.AsyncClient,
    url: str
) -> dict:
    """
    Fetch a public evidence page.

    Redirects are followed manually. Every redirect destination
    is validated before another request is made.
    """

    current_url = url

    for redirect_count in range(MAX_REDIRECTS + 1):

        # Validate every URL, including redirect destinations.
        if not await asyncio.to_thread(is_public_url, current_url):
            return {
                "url": current_url,
                "accessible": False,
                "reason": "URL rejected by public-address validation",
                "text": ""
            }

        try:
            async with client.stream(
                "GET",
                current_url,
                follow_redirects=False
            ) as response:

                # Handle redirects manually.
                if response.is_redirect:

                    location = response.headers.get("location")

                    if not location:
                        return {
                            "url": current_url,
                            "accessible": False,
                            "reason": "Redirect missing location",
                            "text": ""
                        }

                    if redirect_count >= MAX_REDIRECTS:
                        return {
                            "url": current_url,
                            "accessible": False,
                            "reason": "Maximum redirects exceeded",
                            "text": ""
                        }

                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()

                content_type = response.headers.get(
                    "content-type",
                    ""
                ).lower()

                if "text/html" not in content_type:
                    return {
                        "url": current_url,
                        "accessible": False,
                        "reason": "Response is not HTML",
                        "text": ""
                    }

                content = bytearray()

                async for chunk in response.aiter_bytes():

                    content.extend(chunk)

                    if len(content) > MAX_PAGE_SIZE:
                        return {
                            "url": current_url,
                            "accessible": False,
                            "reason": "Page exceeds size limit",
                            "text": ""
                        }

                encoding = response.encoding or "utf-8"

                html = bytes(content).decode(
                    encoding,
                    errors="replace"
                )

                text = extract_page_text(html)

                return {
                    "url": current_url,
                    "accessible": True,
                    "reason": None,
                    "text": text[:20000]
                }

        except Exception as exc:

            logger.warning("Evidence page fetch failed (%s).", type(exc).__name__)

            return {
                "url": current_url,
                "accessible": False,
                "reason": "The evidence page could not be fetched.",
                "text": ""
            }

    return {
        "url": current_url,
        "accessible": False,
        "reason": "Maximum redirects exceeded",
        "text": ""
    }


# --------------------------------------------------
# 4. EVALUATE ACTUAL PAGE EVIDENCE
# --------------------------------------------------

def evaluate_evidence(
    company: dict,
    pages: list[dict],
    analyzed_brand_name: str = "",
    relevance_anchors: list[str] | None = None
) -> dict:
    """
    Evaluate fetched pages for company, product and
    competitor-related evidence.

    Matching is based on extracted page text. An explicit relationship
    requires the page to mention the analyzed brand, the candidate or its
    product, and comparative language in the same sentence. This remains a
    text-matching signal rather than independent proof of market reality.
    """

    company_name = str(
        company.get("name", "")
    ).strip()

    product_name = str(
        company.get("product_name", "")
    ).strip()
    product_name_is_specific = _specific_product_name(company_name, product_name)
    relevance_anchors = [
        str(anchor).strip()
        for anchor in (relevance_anchors or [])
        if str(anchor).strip()
    ]

    if not company_name:
        return {
            "evidence_status": "invalid_company_name",
            "evidence_strength": "none",
            "evidence_details": []
        }

    def normalize_domain(url: str) -> str:
        try:
            hostname = (
                urlsplit(url).hostname or ""
            ).lower()

            return hostname.removeprefix("www.")

        except Exception:
            return ""

    def contains_phrase(text: str, phrase: str) -> bool:
        phrase = phrase.strip().lower()

        if not phrase:
            return False

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(phrase)
            + r"(?![a-z0-9])"
        )

        return bool(re.search(pattern, text.lower()))

    def sentence_matches(text: str, terms: list[str]) -> list[dict]:
        matches = []
        seen = set()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            matching_terms = [
                term for term in terms
                if company_name_matches_text(term, sentence)
            ]
            if matching_terms and sentence not in seen:
                seen.add(sentence)
                matches.append({
                    "matched_text": sentence,
                    "matching_terms": matching_terms
                })
        return matches

    company_domain = normalize_domain(
        company.get("official_website", "")
    )

    def evidence_source_type(domain: str) -> str:
        if not company_domain:
            return "unverified_source"
        return (
            "official_website_candidate"
            if domain == company_domain
            else "third_party"
        )

    generic_test_names = {
        "example",
        "test",
        "sample",
        "demo",
        "company",
    }

    if company_name.lower() in generic_test_names:
        return {
            "evidence_status": "generic_test_name_rejected",
            "evidence_strength": "none",
            "evidence_details": []
        }

    accessible_pages = [page for page in pages if page.get("accessible")]

    # Preserve failure reasons instead of discarding them.
    if not accessible_pages:
        return {
            "evidence_status": "inaccessible",
            "evidence_strength": "none",
            "evidence_details": [
                {
                    "competitor_name": company_name,
                    "url": page.get("url", ""),
                    "source_url": page.get("url", ""),
                    "accessible": False,
                    "accessibility_status": "inaccessible",
                    "classification": "page_inaccessible",
                    "evidence_classification": "page_inaccessible",
                    "is_official_domain": normalize_domain(page.get("url", "")) == company_domain,
                    "source_type": evidence_source_type(
                        normalize_domain(page.get("url", ""))
                    ),
                    "excerpts": [],
                    "reason": page.get("reason", "Unknown error")
                }
                for page in pages
            ]
        }

    details = []
    analyzed_brand_name = str(analyzed_brand_name or "").strip()

    def market_context_matches(anchor: str, sentence: str) -> bool:
        context = sentence
        entity_names = sorted(
            (company_name, product_name, analyzed_brand_name),
            key=lambda name: len(normalize_company_name(name).split()),
            reverse=True
        )
        for entity_name in entity_names:
            tokens = normalize_company_name(entity_name).split()
            if not tokens:
                continue
            pattern = (
                r"(?<![a-z0-9])"
                + r"[\W_]+".join(re.escape(token) for token in tokens)
                + r"(?![a-z0-9])"
            )
            context = re.sub(pattern, " ", context, flags=re.IGNORECASE)
        return _market_relevance_matches(anchor, context)

    for page in pages:

        url = page.get("url", "")
        domain = normalize_domain(url)

        if not page.get("accessible"):

            details.append({
                "competitor_name": company_name,
                "url": url,
                "source_url": url,
                "accessible": False,
                "accessibility_status": "inaccessible",
                "classification": "page_inaccessible",
                "evidence_classification": "page_inaccessible",
                "reason": page.get("reason", "Page unavailable"),
                "company_mentioned": False,
                "product_mentioned": False,
                "competitor_language_found": False,
                "is_official_domain": domain == company_domain,
                "source_type": evidence_source_type(domain),
                "excerpts": []
            })

            continue

        text = page["text"]
        if not text:
            details.append({
                "competitor_name": company_name,
                "url": url,
                "source_url": url,
                "accessible": True,
                "accessibility_status": "accessible",
                "classification": "no_relevant_evidence",
                "evidence_classification": "no_relevant_evidence",
                "company_mentioned": False,
                "product_mentioned": False,
                "competitor_language_found": False,
                "is_official_domain": domain == company_domain,
                "source_type": evidence_source_type(domain),
                "excerpts": []
            })
            continue

        company_mentioned = company_name_matches_text(company_name, text)

        product_mentioned = (
            company_name_matches_text(product_name, text)
            if product_name
            else False
        )

        is_official_domain = bool(company_domain) and domain == company_domain
        relationship_sentences = []
        official_relationship_sentences = []
        negated_relationship_sentences = []
        comparison_sentences = []
        if analyzed_brand_name:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                has_analyzed_brand = company_name_matches_text(analyzed_brand_name, sentence)
                has_candidate = (
                    company_name_matches_text(company_name, sentence)
                    or (product_name and company_name_matches_text(product_name, sentence))
                )
                if not has_analyzed_brand or not has_candidate:
                    continue
                terms = _positive_competitor_terms(sentence)
                if terms:
                    relationship_entry = {
                        "matched_text": sentence.strip(),
                        "matching_terms": list(dict.fromkeys(
                            [term for term in (
                                analyzed_brand_name,
                                company_name,
                                product_name
                            ) if term and contains_phrase(sentence, term)]
                            + terms
                        ))
                    }
                    if is_official_domain:
                        official_relationship_sentences.append(relationship_entry)
                    else:
                        relationship_sentences.append(relationship_entry)
                    continue

                mentioned_terms = _mentioned_competitor_terms(sentence)
                if mentioned_terms:
                    negated_relationship_sentences.append({
                        "matched_text": sentence.strip(),
                        "matching_terms": list(dict.fromkeys(
                            [term for term in (
                                analyzed_brand_name,
                                company_name,
                                product_name
                            ) if term and contains_phrase(sentence, term)]
                            + mentioned_terms
                        ))
                    })
                    continue

                comparison_terms = _positive_comparison_terms(sentence)
                if comparison_terms:
                    comparison_sentences.append({
                        "matched_text": sentence.strip(),
                        "matching_terms": list(dict.fromkeys(
                            [term for term in (
                                analyzed_brand_name,
                                company_name,
                                product_name
                            ) if term and contains_phrase(sentence, term)]
                            + comparison_terms
                        ))
                    })

        competitor_signal = bool(relationship_sentences)
        comparison_signal = bool(comparison_sentences)

        mention_terms = []
        if company_mentioned:
            mention_terms.append(company_name)
        if product_mentioned:
            mention_terms.append(product_name)
        excerpts = sentence_matches(text, mention_terms) if mention_terms else []
        for excerpt in relationship_sentences:
            existing_match = next((
                existing for existing in excerpts
                if existing["matched_text"] == excerpt["matched_text"]
            ), None)
            if existing_match:
                existing_match["matching_terms"] = list(dict.fromkeys(
                    existing_match["matching_terms"] + excerpt["matching_terms"]
                ))
            else:
                excerpts.append(excerpt)
        for excerpt in comparison_sentences:
            existing_match = next((
                existing for existing in excerpts
                if existing["matched_text"] == excerpt["matched_text"]
            ), None)
            if existing_match:
                existing_match["matching_terms"] = list(dict.fromkeys(
                    existing_match["matching_terms"] + excerpt["matching_terms"]
                ))
            else:
                excerpts.append(excerpt)
        for excerpt in negated_relationship_sentences:
            existing_match = next((
                existing for existing in excerpts
                if existing["matched_text"] == excerpt["matched_text"]
            ), None)
            if existing_match:
                existing_match["matching_terms"] = list(dict.fromkeys(
                    existing_match["matching_terms"] + excerpt["matching_terms"]
                ))
            else:
                excerpts.append(excerpt)
        for excerpt in official_relationship_sentences:
            existing_match = next((
                existing for existing in excerpts
                if existing["matched_text"] == excerpt["matched_text"]
            ), None)
            if existing_match:
                existing_match["matching_terms"] = list(dict.fromkeys(
                    existing_match["matching_terms"] + excerpt["matching_terms"]
                ))
            else:
                excerpts.append(excerpt)

        relevant_excerpts = []
        for excerpt in excerpts:
            matched_text = excerpt["matched_text"]
            is_explicit = any(
                item["matched_text"] == matched_text
                for item in relationship_sentences + comparison_sentences
            )
            is_explicit_context = any(
                item["matched_text"] == matched_text
                for item in negated_relationship_sentences + official_relationship_sentences
            )
            is_candidate_product = bool(
                product_name_is_specific
                and company_name_matches_text(product_name, matched_text)
                and (
                    not relevance_anchors
                    or any(
                        market_context_matches(anchor, matched_text)
                        for anchor in relevance_anchors
                    )
                )
            )
            is_candidate_market = (
                not product_name_is_specific
                and company_name_matches_text(company_name, matched_text)
                and any(
                    market_context_matches(anchor, matched_text)
                    for anchor in relevance_anchors
                )
            )
            if is_explicit or is_explicit_context or is_candidate_product or is_candidate_market:
                excerpt["matching_terms"] = list(dict.fromkeys(
                    excerpt["matching_terms"] + [
                        anchor for anchor in relevance_anchors
                        if market_context_matches(anchor, matched_text)
                    ]
                ))
                relevant_excerpts.append(excerpt)
        excerpts = relevant_excerpts

        if competitor_signal:
            page_classification = "explicit_competitor_relationship_found"
        elif comparison_signal:
            page_classification = "product_or_capability_comparison_found"
        elif company_mentioned and product_mentioned and product_name_is_specific and excerpts:
            page_classification = "company_and_product_mentioned"
        elif company_mentioned and excerpts:
            page_classification = "company_mentioned"
        else:
            page_classification = "no_relevant_evidence"
        details.append({
            "competitor_name": company_name,
            "url": url,
            "source_url": url,
            "accessible": True,
            "accessibility_status": "accessible",
            "classification": page_classification,
            "evidence_classification": page_classification,
            "company_mentioned": company_mentioned,
            "product_mentioned": product_mentioned,
            "competitor_language_found": competitor_signal,
            "is_official_domain": is_official_domain,
            "source_type": evidence_source_type(domain),
            "excerpts": [
                _build_excerpt(
                    company_name=company_name,
                    product_name=product_name,
                    url=url,
                    excerpt=excerpt,
                    relationship_sentences=relationship_sentences,
                    comparison_sentences=comparison_sentences,
                    contains_phrase=contains_phrase
                )
                for excerpt in excerpts
            ]
        })

    # Deduplicate identical and near-identical excerpts across fetched pages,
    # keeping the first instance from the already quality-ordered source list.
    seen_excerpt_texts = []
    for detail in details:
        kept_excerpts = []
        omitted = 0
        for excerpt in detail.get("excerpts", []):
            matched_text = excerpt.get("matched_text", "")
            if _is_duplicate_excerpt(matched_text, seen_excerpt_texts):
                omitted += 1
                continue
            kept_excerpts.append(excerpt)
            seen_excerpt_texts.append(matched_text)
        detail["excerpts"] = kept_excerpts
        for excerpt in kept_excerpts:
            excerpt["source_type"] = detail.get("source_type", "third_party")
        if omitted:
            detail["duplicate_excerpts_omitted"] = omitted

    usable_details = [
        item for item in details
        if item.get("accessible")
        and not item.get("reason")
    ]

    if not usable_details:
        return {
            "evidence_status": "no_usable_evidence",
            "evidence_strength": "none",
            "evidence_details": details
        }

    company_support = any(
        item["classification"] in {
            "company_mentioned",
            "company_and_product_mentioned",
        }
        for item in usable_details
    )

    product_support = any(
        item["classification"] == "company_and_product_mentioned"
        for item in usable_details
    )

    explicit_relationship = any(
        item["classification"] == "explicit_competitor_relationship_found"
        for item in usable_details
    )
    explicit_comparison = any(
        item["classification"] == "product_or_capability_comparison_found"
        for item in usable_details
    )

    if explicit_relationship:
        status = "explicit_competitor_relationship_found"
        strength = "strong"

    elif explicit_comparison:
        status = "product_or_capability_comparison_found"
        strength = "moderate"

    elif company_support and product_support:
        status = "market_overlap_unverified"
        strength = "weak"

    elif company_support:
        status = "company_mentioned"
        strength = "weak"

    else:
        status = "no_relevant_evidence"
        strength = "none"

    return {
        "evidence_status": status,
        "evidence_strength": strength,
        "evidence_details": details
    }


def _build_excerpt(
    company_name: str,
    product_name: str,
    url: str,
    excerpt: dict,
    relationship_sentences: list[dict],
    comparison_sentences: list[dict],
    contains_phrase
) -> dict:
    is_relationship_excerpt = any(
        item["matched_text"] == excerpt["matched_text"]
        for item in relationship_sentences
    )
    if is_relationship_excerpt:
        classification = "explicit_competitor_relationship_found"
    elif any(
        item["matched_text"] == excerpt["matched_text"]
        for item in comparison_sentences
    ):
        classification = "product_or_capability_comparison_found"
    elif (
        company_name_matches_text(company_name, excerpt["matched_text"])
        and product_name
        and company_name_matches_text(product_name, excerpt["matched_text"])
    ):
        classification = "company_and_product_mentioned"
    elif company_name_matches_text(product_name, excerpt["matched_text"]) and product_name:
        classification = "product_mentioned"
    else:
        classification = "company_mentioned"
    return {
        "competitor_name": company_name,
        "source_url": url,
        "url": url,
        "matched_text": excerpt["matched_text"],
        "matching_terms": excerpt["matching_terms"],
        "accessible": True,
        "accessibility_status": "accessible",
        "classification": classification,
        "evidence_classification": classification
    }


# --------------------------------------------------
# 5. EVIDENCE CLASSIFICATION
# --------------------------------------------------

def classify_evidence(evidence_status: str) -> dict:
    """
    Assign heuristic evidence confidence and classification.
    These scores are not statistically calibrated probabilities.
    """

    classifications = {

        "explicit_competitor_relationship_found": {
            "evidence_confidence": 0.90,
            "evidence_classification": "explicit_competitor_relationship_found",
            "evidence_strength": "strong"
        },

        "product_or_capability_comparison_found": {
            "evidence_confidence": 0.75,
            "evidence_classification": "product_or_capability_comparison_found",
            "evidence_strength": "moderate"
        },

        "company_and_product_mentioned": {
            "evidence_confidence": 0.60,
            "evidence_classification": "company_and_product_mentioned",
            "evidence_strength": "partial"
        },

        "market_overlap_unverified": {
            "evidence_confidence": 0.35,
            "evidence_classification": "market_overlap_unverified",
            "evidence_strength": "weak"
        },

        "company_mentioned": {
            "evidence_confidence": 0.30,
            "evidence_classification": "company_mentioned",
            "evidence_strength": "weak"
        },

        "no_relevant_evidence": {
            "evidence_confidence": 0.10,
            "evidence_classification": "no_relevant_evidence",
            "evidence_strength": "none"
        },

        "inaccessible": {
            "evidence_confidence": 0.0,
            "evidence_classification": "page_inaccessible",
            "evidence_strength": "none"
        },

        "no_evidence_urls": {
            "evidence_confidence": 0.0,
            "evidence_classification": "no_relevant_evidence",
            "evidence_strength": "none"
        },

        "no_usable_evidence": {
            "evidence_confidence": 0.0,
            "evidence_classification": "page_inaccessible",
            "evidence_strength": "none"
        },

        "generic_test_name_rejected": {
            "evidence_confidence": 0.0,
            "evidence_classification": "rejected",
            "evidence_strength": "none"
        },

        "invalid_company_name": {
            "evidence_confidence": 0.0,
            "evidence_classification": "rejected",
            "evidence_strength": "none"
        },

        "unsupported": {
            "evidence_confidence": 0.0,
            "evidence_classification": "unverified",
            "evidence_strength": "none"
        }
    }

    classification = classifications.get(
        evidence_status,
        {
            "evidence_confidence": 0.0,
            "evidence_classification": "unverified",
            "evidence_strength": "none"
        }
    )
    classification["heuristic_evidence_score"] = classification[
        "evidence_confidence"
    ]
    return classification


# --------------------------------------------------
# 6. VERIFY ONE COMPETITOR
# --------------------------------------------------

async def verify_competitor_evidence(
    company: dict,
    max_pages: int = MAX_PAGES,
    analyzed_brand_name: str = "",
    relevance_anchors: list[str] | None = None
) -> dict:
    """Verify one competitor using third-party and official website pages."""

    evidence_urls = company.get("evidence_urls", [])

    if not isinstance(evidence_urls, list):
        evidence_urls = []

    # Clean and deduplicate evidence URLs.
    all_urls = []
    seen_url_keys = set()
    for raw_url in evidence_urls:
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        url = raw_url.strip()
        key = _evidence_url_key(url)
        if key and key not in seen_url_keys:
            seen_url_keys.add(key)
            all_urls.append(url)

    # Reserve one page slot for the official website when available.
    official_url = company.get("official_website")

    if isinstance(official_url, str) and official_url.strip():
        official_url = official_url.strip()

        official_key = _evidence_url_key(official_url)
        external_urls = [
            url for url in all_urls
            if _evidence_url_key(url) != official_key
        ]
        external_urls.sort(key=_source_priority)

        if max_pages >= 1:
            urls = external_urls[:max_pages - 1]
            urls.insert(0, official_url)
        else:
            urls = []
    else:
        urls = sorted(all_urls, key=_source_priority)[:max_pages]

    if not urls:
        status = "no_evidence_urls"
        classification = classify_evidence(status)

        return {
            "name": company.get("name", ""),
            "product_name": company.get("product_name", ""),
            "evidence_status": status,
            **classification,
            "evidence_pages_checked": 0,
            "evidence_pages_accessible": 0,
            "evidence_details": []
        }

    timeout = httpx.Timeout(REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; LinklyAI/1.0; "
                "+https://example.com/bot)"
            )
        },
        trust_env=False
    ) as client:
        pages = await asyncio.gather(
            *[
                fetch_evidence_page(client, url)
                for url in urls
            ]
        )

    evaluation = evaluate_evidence(
        company,
        pages,
        analyzed_brand_name=analyzed_brand_name,
        relevance_anchors=relevance_anchors
    )
    status = evaluation["evidence_status"]
    classification = classify_evidence(status)

    return {
        "name": company.get("name", ""),
        "product_name": company.get("product_name", ""),
        "evidence_status": status,
        **classification,
        "evidence_pages_checked": len(pages),
        "evidence_pages_accessible": sum(
            1 for page in pages
            if page.get("accessible")
        ),
        "evidence_details": evaluation["evidence_details"]
    }


# --------------------------------------------------
# 7. VERIFY ALL COMPETITORS
# --------------------------------------------------

async def verify_competitors(
    competitors: list[dict],
    analyzed_brand_name: str = "",
    relevance_anchors: list[str] | None = None
) -> list[dict]:

    if not isinstance(competitors, list):
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CANDIDATES)

    async def verify_one(competitor: dict):
        async with semaphore:
            return await verify_competitor_evidence(
                competitor,
                analyzed_brand_name=analyzed_brand_name,
                relevance_anchors=relevance_anchors
            )

    results = await asyncio.gather(
        *[
            verify_one(competitor) if isinstance(competitor, dict)
            else asyncio.sleep(0, result=ValueError("Invalid candidate shape"))
            for competitor in competitors
        ],
        return_exceptions=True
    )

    verified_results = []
    candidate_error_count = 0

    for competitor, result in zip(competitors, results):

        if isinstance(result, Exception):
            candidate_error_count += 1

            error_code = getattr(result, "code", None) or getattr(result, "status_code", None)
            if error_code is not None:
                error_code = re.sub(r"[^A-Za-z0-9_-]", "", str(error_code))[:64] or None

            logger.warning(
                "Candidate evidence verification failed (%s).",
                type(result).__name__,
                extra={
                    "diagnostic_event": "competitor_evidence.candidate_failed",
                    "error_type": type(result).__name__,
                    "error_code": error_code
                }
            )

            result = {
                "name": competitor.get("name", "") if isinstance(competitor, dict) else "",
                "evidence_status": "verification_error",
                "evidence_confidence": 0.0,
                "heuristic_evidence_score": 0.0,
                "evidence_classification": "unverified",
                "evidence_strength": "none",
                "evidence_pages_checked": 0,
                "evidence_pages_accessible": 0,
                "evidence_details": []
            }

        verified_results.append({
            **(competitor if isinstance(competitor, dict) else {}),
            **result
        })

    pages_checked = sum(int(item.get("evidence_pages_checked", 0) or 0) for item in verified_results)
    pages_accessible = sum(int(item.get("evidence_pages_accessible", 0) or 0) for item in verified_results)
    logger.info(
        "competitor_evidence.verification_batch_completed "
        "candidate_count=%d page_fetch_success_count=%d "
        "page_fetch_inaccessible_count=%d candidate_error_count=%d",
        len(competitors), pages_accessible, max(0, pages_checked - pages_accessible),
        candidate_error_count,
        extra={
            "diagnostic_event": "competitor_evidence.verification_batch_completed",
            "candidate_count": len(competitors),
            "page_fetch_success_count": pages_accessible,
            "page_fetch_inaccessible_count": max(0, pages_checked - pages_accessible),
            "candidate_error_count": candidate_error_count
        }
    )

    return verified_results
