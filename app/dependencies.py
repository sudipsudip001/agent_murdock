from rerankers import Reranker

reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = Reranker(reranker_name, device="cpu")
