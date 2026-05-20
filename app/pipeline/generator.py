import logging
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.documents import Document

from app.models.response import Citation, GenResponse, RAGResponse, TokenUsage

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Generator:
    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        self.model = model
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={"timeout": 30000},
        )
        self.system_prompt = """
            Role: You are a helpful assistant that answers questions using ONLY the provided context.
            Rules:
                - Cite sources inline using [1], [2], etc. after every claim.
                - Only include sources you actually cited inline.
                - Synthesize information in your own words.
                - citations must be ordered by citation number.
                - If the context lacks enough information, say so in the answer field.
                - If the answer isn't present in the context, set answer to: THE ANSWER COULDN'T BE FOUND IN THE CONTEXT.
            The JSON must have exactly this structure:
            {
                "answer": "your answer with inline citations like [1], [2]",
                "citations": [
                    {"src": "source", "page": page_num},
                    {"src": "source", "page": page_num},
                ]
            }
        """

    def generate_answer(
        self,
        context_list: list[Document],
        query: str,
    ) -> GenResponse:
        user_prompt = f"Context:\n{context_list}\n\nQuestion:{query}"
        response = self.client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=RAGResponse,
            ),
            contents=[user_prompt],
        )
        parsed_response = response.parsed
        logger.debug(f"THE RESPONSE OF THE MODEL: {parsed_response}")
        usage = response.usage_metadata
        return GenResponse(
            response=parsed_response.answer,
            citations={
                str(i + 1): Citation(src=p.src, page=p.page)
                for i, p in enumerate(parsed_response.citations)
            },
            token_use=TokenUsage(
                prompt_token_count=getattr(usage, "prompt_token_count", 0),
                total_token_count=getattr(usage, "total_token_count", 0),
            ),
        )
