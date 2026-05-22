from langchain_core.documents import Document

from app.dependencies import reranker


class Ranker:
    def __init__(
        self,
    ) -> None:
        pass

    def reranked_docs(
        self,
        initial_docs: list[Document],
        query: str,
        num_final_docs: int = 3,
    ) -> list[Document]:
        print("===> Reranking documents...")
        doc_texts = [doc.page_content for doc in initial_docs]
        rerank_results = reranker.rank(query=query, docs=doc_texts)
        reranked_docs = []
        for res in rerank_results.results[:num_final_docs]:
            doc = initial_docs[res.doc_id]
            doc.metadata["rerank_score"] = res.score
            reranked_docs.append(doc)
        return reranked_docs
