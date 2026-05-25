from rerankers import Reranker

from app.pipeline.agent.graph import Graph

reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = Reranker(reranker_name, device="cpu")
graph = Graph()
