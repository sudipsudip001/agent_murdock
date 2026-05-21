from typing import TypedDict

from langchain_core.documents import Document

from app.models.response import GenResponse


class RAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: GenResponse | None
    iterations: int
