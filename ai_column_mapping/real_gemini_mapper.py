import json
import re
from google import genai
from google.api_core.exceptions import ResourceExhausted
from .base_column_mapper import BaseColumnMapper


class GeminiGenAIColumnMapper(BaseColumnMapper):
    """
    Safe GenAI mapper with automatic fallback on quota / API failure
    """

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-2.5-flash"

    def map_columns(self, source_columns: list, CANONICAL_schema: dict) -> dict:
        prompt = self._build_prompt(source_columns, CANONICAL_schema)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            json_text = self._extract_json(response.text)
            ai_mapping = json.loads(json_text)

        except ResourceExhausted:
            print(" Gemini quota exceeded — switching to fallback mapping")
            return {col: None for col in source_columns}

        except Exception as e:
            print(f" Gemini failure — fallback mapping used: {e}")
            return {col: None for col in source_columns}

        # Ensure every source column exists
        return {col: ai_mapping.get(col) for col in source_columns}

    def _build_prompt(self, source_columns, CANONICAL_schema):
        return f"""
You are a senior data engineer.

Map source columns to canonical schema fields using semantic meaning.

Rules:
- Always map if meaning matches
- Use null ONLY if completely unrelated
- Output JSON only
- Keys must exactly match source columns

Canonical schema:
{list(CANONICAL_schema.keys())}

Source columns:
{source_columns}
"""

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in Gemini response")
        return match.group(0)
