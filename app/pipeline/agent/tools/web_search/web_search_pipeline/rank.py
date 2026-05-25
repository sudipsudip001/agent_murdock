from rerankers import Reranker


class Rank:
    def __init__(
        self,
    ) -> None:
        self._reranker = None

    def reranked_docs(
        self,
        reranker: Reranker,
        initial_docs: list[dict[str, str]],
        query: str,
        num_final_docs: int = 3,
    ) -> list[dict[str, str]]:
        print("===> Reranking documents...")

        doc_texts = [doc["text"] for doc in initial_docs]

        rerank_results = reranker.rank(query=query, docs=doc_texts)

        reranked_docs = []
        for res in rerank_results.results[:num_final_docs]:
            original_dict = initial_docs[res.doc_id]
            original_dict["rerank_score"] = res.score

            reranked_docs.append(original_dict)
        return reranked_docs
