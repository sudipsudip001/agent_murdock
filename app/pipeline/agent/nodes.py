from typing import Any

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from app.pipeline.agent.state import RAGState
from app.pipeline.ranker import Ranker
from app.pipeline.weaver import Weaver


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

    ### USE THE GENERATOR FROM YOUR NORMAL RAG INTERFACE.
    def generate(self, state: RAGState) -> dict[str, Any]:
        context = "\n\n---\n\n".join(doc.page_content for doc in state["documents"])
        prompt = f"""Answer the question using only the context below.
        If the context doesn't contain enough information, say so clearly.
        Context:
        {context}
        Question: {state['question']}"""
        answer = self.llm.invoke(prompt).content
        return {"generation": answer}

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
