

import asyncio
import json

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL


# Initialize Gemini client
if not GEMINI_API_KEY:
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)


async def analyze_with_ai(website_data: dict):
    """
    Analyze extracted website information using Gemini.
    Returns structured business, competitor, SEO and marketing insights.
    """

    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your environment variables."
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
6. Use Google, Facebook, Instagram, LinkedIn and X as
   search platforms.
7. Use each of these five platforms exactly once in the
   five target_search_queries. Do not repeat platforms.
8. Do not claim that a query has actual search volume or
   verified popularity unless reliable data is provided.
9. Use extracted website content as the primary source.
10. Mark inferred services with inferred=true.
11. Clearly distinguish verified website facts from
    suggestions and AI-generated recommendations.
12. If information is unavailable, use "Not available".
13. Infer relevant industry segments only when reasonable.
14. Clearly label inferred segments with "(Inferred)".
15. Do not present inferred information as verified facts.
16. Keep recommendations practical, relevant and specific
    to the analyzed brand.
17. Do not repeat irrelevant or unrelated website content.
18. Ensure all five search queries contain query, platform,
    intent, relevance and target_audience.
19. Ensure all five services contain name, description,
    target_industry, target_customer, business_benefit
    and inferred.
20. Return valid JSON only. Do not include Markdown fences,
    introductory text, explanations or comments.
21. For competitor strengths and weaknesses, begin each
    item with either "Verified:" or
    "AI-generated suggestion:".
22. Use "Verified:" only when the website data or reliable
    supplied evidence supports the claim.
23. Use "AI-generated suggestion:" for assumptions or
    suggestions that have not been independently verified.
24. Set inferred=true if a service's target industry or
    target customer is inferred rather than explicitly
    supported by the website content.
25. Set inferred=false only when the relevant information
    is explicitly supported by the website content.
26. Do not confuse products listed on the website with
    services that the business actually provides.

Follow the exact JSON structure above.
"""

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            result = json.loads(response.text)

            # Validate top-level structure
            required_sections = {
                "competitor_analysis",
                "target_search_queries",
                "seo_analysis",
                "top_services",
                "target_audience"
            }

            if not isinstance(result, dict):
                raise ValueError(
                    "Gemini response must be a JSON object."
                )

            missing = required_sections - result.keys()

            if missing:
                raise ValueError(
                    f"Missing required sections: {', '.join(missing)}"
                )

            # Validate search queries
            queries = result["target_search_queries"]

            if not isinstance(queries, list) or len(queries) != 5:
                raise ValueError(
                    "Gemini must return exactly 5 search queries."
                )

            required_platforms = {
                "google",
                "facebook",
                "instagram",
                "linkedin",
                "x"
            }

            actual_platforms = {
                str(item.get("platform", "")).strip().lower()
                for item in queries
                if isinstance(item, dict)
            }

            if len(actual_platforms) != 5:
                raise ValueError(
                    "Each search query must use a different platform."
                )

            if actual_platforms != required_platforms:
                raise ValueError(
                    "Search queries must cover Google, Facebook, "
                    "Instagram, LinkedIn and X exactly once."
                )

            query_fields = {
                "query",
                "platform",
                "intent",
                "relevance",
                "target_audience"
            }

            for item in queries:
                if not isinstance(item, dict):
                    raise ValueError("Invalid search query format.")

                missing_fields = query_fields - item.keys()

                if missing_fields:
                    raise ValueError(
                        f"Search query missing fields: {missing_fields}"
                    )

            # Validate top services
            services = result["top_services"]

            if not isinstance(services, list) or len(services) != 5:
                raise ValueError(
                    "Gemini must return exactly 5 top services."
                )

            service_fields = {
                "name",
                "description",
                "target_industry",
                "target_customer",
                "business_benefit",
                "inferred"
            }

            for service in services:
                if not isinstance(service, dict):
                    raise ValueError("Invalid service format.")

                missing_fields = service_fields - service.keys()

                if missing_fields:
                    raise ValueError(
                        f"Service missing fields: {missing_fields}"
                    )

                if not isinstance(service["inferred"], bool):
                    raise ValueError(
                        "Service inferred field must be true or false."
                    )

            # Validate competitor analysis
            competitor_data = result["competitor_analysis"]

            if not isinstance(competitor_data, dict):
                raise ValueError(
                    "Invalid competitor_analysis format."
                )

            if not isinstance(competitor_data.get("competitors"), list):
                raise ValueError(
                    "Competitors must be returned as a list."
                )

            # Validate SEO and audience sections
            if not isinstance(result["seo_analysis"], dict):
                raise ValueError("Invalid SEO analysis format.")

            if not isinstance(result["target_audience"], dict):
                raise ValueError("Invalid target audience format.")

            return result

        except Exception as error:
            error_message = str(error).upper()

            temporary_error = any(
                term in error_message
                for term in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "INTERNAL",
                    "SEARCH QUERIES",
                    "PLATFORMS",
                    "MISSING FIELDS",
                    "MUST RETURN"
                ]
            )

            if temporary_error and attempt < max_attempts - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue

            raise ValueError(
                f"Gemini analysis failed: {error}"
            ) from error

