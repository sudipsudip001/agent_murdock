from langchain_core.documents import Document
from langchain_core.tools import tool

from app.pipeline.agent.tools.vector_db.vector_creator.ranker import Ranker
from app.pipeline.weaver import Weaver

weaver = Weaver()
ranker = Ranker()


@tool  # type: ignore[misc]
def retrieve(query: str) -> dict[str, list[Document]]:
    """
    Search the knowledge base for documents relevant to the query.
    Use this whenever you need to find information to answer a question.
    Always try this first before concluding information is unavailable.
    """
    sim_docs = weaver.return_similar_docs(
        query=query,
    )
    docs = ranker.reranked_docs(
        initial_docs=sim_docs,
        query=query,
        num_final_docs=3,
    )
    return {"documents": docs}
