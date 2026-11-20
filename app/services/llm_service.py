

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
    Returns structured business, SEO and marketing insights.
    """

    if client is None:
        raise ValueError(
            "GEMINI_API_KEY is missing. Check your .env file."
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


STRICT RULES:

1. Return exactly 5 target_search_queries.
2. Return exactly 5 top_services.
3. Include at least 3 relevant competitors when reliable
   information is available. Do not invent competitors.
4. Never invent search volumes, statistics, percentages,
   rankings, customer counts, revenue figures or factual claims.
5. Never generate numerical statistics, percentages,
   rankings or performance claims unless explicitly
   supported by the extracted website content.
6. Use Google, Facebook, Instagram, LinkedIn and X as
   search platforms. Distribute queries across platforms
   where relevant. Google may be used more than once.
7. Do not claim that a query has actual search volume or
   verified popularity unless reliable data is provided.
8. Use extracted website content as the primary source.
9. Mark inferred services with inferred=true.
10. Clearly distinguish verified website facts from
    suggestions and AI-generated recommendations.
11. If information is unavailable, use "Not available".
12. Infer relevant industry segments only when reasonable.
    Clearly label inferred segments with "(Inferred)".
13. Do not present inferred information as verified facts.
14. Keep recommendations practical, relevant and specific
    to the analyzed brand.
15. Do not repeat irrelevant or unrelated website content.
16. Ensure all five search queries contain a query, platform,
    intent, relevance and target_audience.
17. Ensure all five services contain a name, description,
    target_industry, target_customer, business_benefit
    and inferred field.
18. Return valid JSON only.
19. Do not include Markdown code fences, introductory text,
    explanations or comments outside the JSON object.
20. Follow the exact JSON structure specified above.
"""

    # Retry temporary Gemini availability errors
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

            required_sections = {
                "competitor_analysis",
                "target_search_queries",
                "seo_analysis",
                "top_services",
                "target_audience"
            }

            if not isinstance(result, dict):
                raise ValueError("Gemini response must be a JSON object.")

            missing = required_sections - result.keys()

            if missing:
                raise ValueError(
                    f"Missing required sections: {', '.join(missing)}"
                )

            if len(result["target_search_queries"]) != 5:
                raise ValueError(
                    "Gemini must return exactly 5 search queries."
                )

            if len(result["top_services"]) != 5:
                raise ValueError(
                    "Gemini must return exactly 5 services."
                )

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
                    "INTERNAL"
                ]
            )

            if temporary_error and attempt < max_attempts - 1:
                await asyncio.sleep(2 ** (attempt + 1))
                continue

            raise ValueError(
                f"Gemini analysis failed: {error}"
            ) from error

