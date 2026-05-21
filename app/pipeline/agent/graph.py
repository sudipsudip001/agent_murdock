from typing import cast

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.models.response import GenResponse
from app.pipeline.agent.nodes import Nodes
from app.pipeline.agent.state import RAGState


class Graph:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.nodes = Nodes(llm=llm)
        self.app = self._build()

    def _build(self) -> StateGraph:
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self.nodes.retrieve)
        workflow.add_node("grade_documents", self.nodes.grade_documents)
        workflow.add_node("generate", self.nodes.generate)
        workflow.add_node("rewrite_query", self.nodes.rewrite_query)

        workflow.set_entry_point("retrieve")

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("rewrite_query", "retrieve")

        workflow.add_conditional_edges(
            "grade_documents",
            self.nodes.decide_after_grading,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            self.nodes.decide_after_generation,
            {"end": END},
        )
        return workflow.compile()

    def run(self, question: str) -> GenResponse:
        result = self.app.invoke(
            {
                "question": question,
                "documents": [],
                "generation": None,
                "iterations": 0,
            }
        )
        return cast(GenResponse, result["generation"])
