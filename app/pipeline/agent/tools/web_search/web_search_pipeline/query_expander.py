from datetime import date

from fastapi import HTTPException
from google import genai
from google.api_core import exceptions
from google.genai import types
from pydantic import BaseModel

from app.config import GEMINI_API_KEY


class QueryExpansion(BaseModel):
    question_type: str
    questions: list[str]


class QueryExpander:
    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self._raw_prompt = """
            Role: You are an expert search query optimizer.
            Task: Given the user query, identify it's type and use the guidelines to reformulate questions for it.
            Today's date is {current_date}.

            Guidelines for query types:
            - MULTIFACETED/COMPARISON: For Multifaceted queries, produce multiple independent sub-queries relating to each topic
            - FACTUAL: Inject known entities or date as per requirement
            - EXPLORATORY: Generate the answer to the user query first, and also generate a more general, principled version of the query to retrieve background context
            - NAVIGATIONAL: Just search the direct answers.

            Constraints:
            - Do NOT drift from the original topic.
            - Generate as many contextual questions as you want so that answering them will help answer the original user question.
            - Use concise, keyword-rich, search-engine-friendly language. No full sentences or filler words.
            - If the original query implies or needs a date (e.g., "current", "latest", "2026"), include appropriate date markers.
            - Output strictly a JSON object with keys "question1", "question2", .. No additional text.
        """
        self.model = model
        self.client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": 30000},
        )

    def expanded_queries(self, query: str) -> list[str]:
        prompt = self._raw_prompt.format(current_date=date.today().strftime("%Y-%m-%d"))
        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=prompt,
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=QueryExpansion,
                ),
                contents=f"User Query: {query}",
            )
            expanded = QueryExpansion.model_validate_json(response.text)
            return [q for q in expanded.questions if q is not None]
        except exceptions.ResourceExhausted as e:
            raise HTTPException(
                status_code=429,
                detail="Quota exceeded! Please check AI studio billing.",
            ) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
