
import asyncio
import json
import re

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL


# Initialize Gemini client
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


def normalize_inferred_fields(services: list) -> list:
    """
    Ensure inferred=True whenever the industry or customer
    segment is explicitly marked as inferred.
    """
    for service in services:
        industry = str(service.get("target_industry", ""))
        customer = str(service.get("target_customer", ""))

        is_inferred = (
            "(inferred)" in industry.lower()
            or "(inferred)" in customer.lower()
        )

        service["inferred"] = is_inferred

        # Keep the inferred label consistent with the Boolean.
        if is_inferred:
            if "(Inferred)" not in industry:
                service["target_industry"] = f"{industry} (Inferred)"

    return services


def validate_analysis(result: dict) -> dict:
    """
    Validate Gemini's JSON response and required sections.
    """

    required_sections = {
        "competitor_analysis",
        "target_search_queries",
        "seo_analysis",
        "top_services",
        "target_audience",
    }

    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object.")

    missing = required_sections - result.keys()

    if missing:
        raise ValueError(
            f"Missing required sections: {', '.join(sorted(missing))}"
        )

    # Validate competitor analysis
    competitor = result["competitor_analysis"]

    if not isinstance(competitor, dict):
        raise ValueError("Invalid competitor analysis format.")

    required_competitor_fields = [
        "brand_name",
        "overview",
        "category",
        "products_services",
        "competitors",
        "brand_advantages",
        "positioning_opportunities",
    ]

    for field in required_competitor_fields:
        if field not in competitor:
            raise ValueError(
                f"Missing competitor analysis field: {field}"
            )

    if not isinstance(competitor["competitors"], list):
        raise ValueError("Competitors must be a list.")

    # Validate target search queries
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
        "target_audience",
    ]

    allowed_platforms = {
        "Google",
        "Facebook",
        "Instagram",
        "LinkedIn",
        "X",
    }

    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Each search query must be an object.")

        for field in required_query_fields:
            if not query.get(field):
                raise ValueError(
                    f"Search query is missing field: {field}"
                )

        if query["platform"] not in allowed_platforms:
            raise ValueError(
                f"Unsupported search platform: {query['platform']}"
            )

    platforms = {item["platform"] for item in queries}

    if platforms != allowed_platforms:
        raise ValueError(
            "Search queries must cover Google, Facebook, Instagram, "
            "LinkedIn and X."
        )

    # Validate SEO analysis
    seo = result["seo_analysis"]

    if not isinstance(seo, dict):
        raise ValueError("Invalid SEO analysis format.")

    required_seo_fields = [
        "primary_keywords",
        "secondary_keywords",
        "long_tail_keywords",
        "business_keywords",
        "seo_title",
        "meta_description",
        "content_recommendations",
    ]

    for field in required_seo_fields:
        if field not in seo:
            raise ValueError(f"Missing SEO field: {field}")

    # Validate top services
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
        "inferred",
    ]

    for service in services:
        if not isinstance(service, dict):
            raise ValueError("Each service must be an object.")

        # Add a safe default if Gemini omits target_customer.
        if not service.get("target_customer"):
            service["target_customer"] = "Not available"

        # Default inferred to False when the model omits the flag.
        if "inferred" not in service:
            service["inferred"] = False

        for field in required_service_fields:
            if field not in service:
                raise ValueError(
                    f"Service is missing field: {field}"
                )

    # Fix inferred labels and Boolean values.
    result["top_services"] = normalize_inferred_fields(services)

    # Validate target audience
    audience = result["target_audience"]

    if not isinstance(audience, dict):
        raise ValueError("Invalid target audience format.")

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
    ]

    for field in required_audience_fields:
        if field not in audience:
            raise ValueError(
                f"Missing target audience field: {field}"
            )

    return result


async def analyze_with_ai(website_data: dict) -> dict:
    """
    Analyze extracted website information using Google Gemini.

    Returns structured competitor, search query, SEO,
    services and target audience analysis.
    """

    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing. Configure it in your environment."
        )

    prompt = f"""
You are an expert business analyst, competitor researcher,
SEO strategist and digital marketing consultant.

Analyze the following extracted website information.

WEBSITE DATA:
{json.dumps(website_data, ensure_ascii=False)}

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

STRICT RULES:

1. Return exactly 5 target_search_queries.
2. Return exactly 5 top_services.
3. Include at least 3 relevant competitors when reliable
   information is available. Do not invent competitors.
4. Never invent search volumes, statistics, percentages,
   rankings, customer counts, revenue figures or factual claims.
5. Never generate numerical statistics unless explicitly
   supported by the extracted website content.
6. Use these search platforms across the five queries:
   Google, Facebook, Instagram, LinkedIn and X.
   Use each platform exactly once.
7. Do not claim that a query has actual search volume or
   verified popularity unless reliable data is provided.
8. Use extracted website content as the primary source.
9. Clearly distinguish verified website facts from AI-generated
   suggestions and recommendations.
10. Mark inferred services with inferred=true.
11. If the industry or customer segment is inferred,
    append "(Inferred)" to the relevant field.
12. Never present inferred information as verified facts.
13. If information is unavailable, use "Not available".
14. Infer relevant industry segments only when reasonable.
    Clearly label inferred segments with "(Inferred)".
15. Keep recommendations practical, relevant and specific
    to the analyzed brand.
16. Do not repeat irrelevant or unrelated website content.
17. Ensure all five search queries contain query, platform,
    intent, relevance and target_audience.
18. Ensure all five services contain name, description,
    target_industry, target_customer, business_benefit
    and inferred.
19. Return valid JSON only.
20. Do not include Markdown code fences, introductory text,
    explanations or comments outside the JSON object.
21. Never invent statistics, percentages, customer counts,
    revenue figures, performance claims or numerical benefits.

22. Only include numerical claims when explicitly supported
    by the extracted website content.

23. Competitor strengths and weaknesses must be clearly
    identified as AI-generated assessments unless supported
    by reliable evidence in the provided website data.

24. Do not present competitor assessments as verified facts
    when external competitor research has not been performed.

25. If reliable competitor information is unavailable,
    explicitly state "Not verified from provided website data".

26. Do not invent competitor pricing, market share,
    rankings or customer numbers.
27. Use clear, concise geographic descriptions.
28. Do not use redundant phrases such as "Global Worldwide".
29. Use "Worldwide" only when the website supports a global
    geographic presence. Otherwise, use the specific regions
    mentioned on the website.
30. If geographic information is unavailable, return
    "Not available".
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            response_text = response.text.strip()

            # Remove accidental Markdown code fences
            response_text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                response_text,
                flags=re.IGNORECASE,
            ).strip()

            result = json.loads(response_text)

            return validate_analysis(result)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"Gemini returned invalid JSON: {error}"
            ) from error

        except ValueError:
            raise

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
                ]
            )

            if temporary_error and attempt < max_attempts - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue

            raise ValueError(
                f"Gemini analysis failed: {error}"
            ) from error

    raise ValueError("Gemini analysis failed after multiple attempts.")


