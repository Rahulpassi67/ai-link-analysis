

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


# ============================================================
# 5. GEMINI RESPONSE VALIDATION
# ============================================================

def validate_analysis(result: dict) -> dict:
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

    queries = result["target_search_queries"]

    if not isinstance(queries, list) or len(queries) != 5:
        raise ValueError(
            "Gemini must return exactly 5 search queries."
        )

    required_query_fields = [
        "query",
        "platform",
        "intent",
        "relevance",
        "target_audience"
    ]

    allowed_platforms = {
        "Google",
        "Facebook",
        "Instagram",
        "LinkedIn",
        "X"
    }

    for query in queries:

        validate_required_fields(
            query,
            required_query_fields,
            "search query"
        )

        for field in required_query_fields:

            if not isinstance(query[field], str) or not query[field].strip():
                raise ValueError(
                    f"Search query field {field} must be a non-empty string."
                )

        if query["platform"] not in allowed_platforms:
            raise ValueError(
                f"Unsupported search platform: "
                f"{query['platform']}"
            )

    platforms = {
        item["platform"] for item in queries
    }

    if platforms != allowed_platforms:
        raise ValueError(
            "Search queries must cover Google, Facebook, "
            "Instagram, LinkedIn and X exactly once."
        )

    # --------------------------------------------------------
    # SEO ANALYSIS
    # --------------------------------------------------------

    seo = result["seo_analysis"]

    required_seo_fields = [
        "primary_keywords",
        "secondary_keywords",
        "long_tail_keywords",
        "business_keywords",
        "seo_title",
        "meta_description",
        "content_recommendations"
    ]

    validate_required_fields(
        seo,
        required_seo_fields,
        "SEO analysis"
    )

    for field in [
        "primary_keywords",
        "secondary_keywords",
        "long_tail_keywords",
        "business_keywords",
        "content_recommendations"
    ]:
        validate_string_list(
            seo[field],
            field
        )

    for field in [
        "seo_title",
        "meta_description"
    ]:
        if not isinstance(seo[field], str):
            raise ValueError(
                f"SEO field {field} must be a string."
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

    required_audience_fields = [
        "primary_audience",
        "secondary_audience",
        "industry_segments",
        "business_types",
        "customer_profiles",
        "geographic_focus",
        "needs_and_challenges",
        "interests",
        "recommended_marketing_segments"
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
    "seo_title": "",
    "meta_description": "",
    "content_recommendations": []
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
    "business_types": [],
    "customer_profiles": [],
    "geographic_focus": [],
    "needs_and_challenges": [],
    "interests": [],
    "recommended_marketing_segments": []
  }}
}}

STRICT OUTPUT RULES:

1. Return exactly 5 target_search_queries.

2. Return exactly 5 top_services.

3. Use these platforms across the five queries:
   Google, Facebook, Instagram, LinkedIn and X.
   Use each exactly once.

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
"""

    # ========================================================
    # 7. GEMINI API WITH RETRIES
    # ========================================================

    max_attempts = 3

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

            if not response.text:
                raise ExternalServiceError(
                    "gemini_empty_response",
                    "Gemini returned an empty analysis response.",
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
                raise ExternalServiceError(
                    "gemini_invalid_response",
                    "Gemini returned an invalid analysis response. Please try again.",
                    502
                ) from error

            # Validate all sections and fields.
            try:
                validated_result = validate_analysis(result)
            except ValueError as error:
                raise ExternalServiceError(
                    "gemini_invalid_response",
                    "Gemini returned an incomplete analysis response. Please try again.",
                    502
                ) from error

            # Add ML-based competitor ranking.
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
