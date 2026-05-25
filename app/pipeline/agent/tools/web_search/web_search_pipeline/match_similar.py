from typing import cast

from rank_bm25 import BM25Okapi


class MatchSimilar:
    def __init__(
        self,
        chunked_docs: list[dict[str, str]],
    ) -> None:
        self.chunked_docs = chunked_docs
        self.tokenized_corpus = [
            (doc["title"] + " " + doc["text"]).lower().split()
            for doc in self.chunked_docs
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def match_similar_docs(
        self,
        query: str,
        num_docs: int = 5,
    ) -> list[dict[str, str]]:
        tokenized_query = query.lower().split()
        top_n_results = self.bm25.get_top_n(
            tokenized_query, self.chunked_docs, n=num_docs
        )

        return cast(list[dict[str, str]], top_n_results)
