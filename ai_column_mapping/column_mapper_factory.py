from ai_column_mapping.mock_ai_mapper import MockGenAIColumnMapper
from ai_column_mapping.real_gemini_mapper import GeminiGenAIColumnMapper
from .mock_ai_mapper import MockGenAIColumnMapper
 

def get_column_mapper(use_real_genai=False, api_key=None):
 
    if use_real_genai:
        if not api_key:
            raise ValueError("Gemini API key required for real GenAI")
        return GeminiGenAIColumnMapper(api_key)
    return MockGenAIColumnMapper()


