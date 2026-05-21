import logging
import os
from typing import Any

from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models.response import Citation, GenResponse, RAGResponse, TokenUsage
from app.pipeline.agent.state import RAGState
from app.pipeline.ranker import Ranker
from app.pipeline.weaver import Weaver

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Nodes:
    def __init__(self, llm: ChatGoogleGenerativeAI) -> None:
        self.weaver = Weaver()
        self.ranker = Ranker()
        self.llm = llm

    def retrieve(
        self,
        state: RAGState,
    ) -> dict[str, Document]:
        """Drop-in: calls your existing Weaviate retreiver (with reranking)."""
        sim_docs = self.weaver.return_similar_docs(
            query=state["question"],
        )
        docs = self.ranker.reranked_docs(
            initial_docs=sim_docs,
            query=state["question"],
            num_final_docs=3,
        )
        return {"documents": docs}

    def grade_documents(
        self,
        state: RAGState,
    ) -> dict[str, Document]:
        """Ask the LLM whether each doc is relevant. Filter the irrelevant ones out."""
        question = state["question"]
        filtered = []
        for doc in state["documents"]:
            prompt = f"""
                You are a document relevance classifier.
                Is the following document relevant to the question?
                STRICLY Reply only with 'yes' or 'no'.

                Question: {question}
                Document: {doc.page_content}
            """
            result = self.llm.invoke(prompt).content.strip().lower()
            if "yes" in result:
                filtered.append(doc)
        return {"documents": filtered}

    def rewrite_query(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        """Called when retrieved docs were mostly irrelevant. Gemini
        rewrites the question to improve the next retrieval.
        """
        prompt = f"""
            The query below failed to retrieve useful documents.
            Rewrite it to be more specific and retrieval-friendly.
            Return only the rewritten question, nothing else.

            Original question: {state["question"]}
        """
        new_question = self.llm.invoke(prompt).content.strip()
        return {"question": new_question, "iterations": state.get("iterations", 0) + 1}

    def generate(self, state: RAGState) -> dict[str, GenResponse]:
        system_prompt = """
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
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": 50000},
        )
        context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
        user_prompt = f"Context:\n{context}\n\nQuestion: {state["question"]}"
        logger.debug(f"THE FINAL USER PROMPT IS GIVEN AS: {user_prompt}")

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=RAGResponse,
            ),
            contents=[user_prompt],
        )
        parsed_response = response.parsed
        usage = response.usage_metadata

        logger.debug(f"The final parsed response is given as: {parsed_response}")
        gen_response = GenResponse(
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

        return {"generation": gen_response}

    # --4. Conditional Edge Logic-----------------------------------
    def decide_after_grading(
        self,
        state: RAGState,
    ) -> str:
        """After grading, decide: do we have enough docs to generate?
        If not -> rewrite the query and try again."""
        if state.get("iterations", 0) >= 3:
            return "generate"
        if len(state["documents"]) == 0:
            return "rewrite_query"
        return "generate"

    def decide_after_generation(
        self,
        state: RAGState,  # noqa: ARG002
    ) -> str:
        """Optionally grade answer itself."""
        return "end"
