import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from app.models.response import Context


class Chunker:
    MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embedder_model: str = "nomic-ai/nomic-embed-text-v1",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EMBEDDING_MODEL_NAME = embedder_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )

    def chunk_documents(
        self,
        knowledge_base: list[Context],
    ) -> list[dict[str, str]]:
        seen, chunks = set(), []
        for item in knowledge_base:
            texts = self.text_splitter.split_text(item.text)
            for t in texts:
                if t not in seen:
                    seen.add(t)
                    chunks.append({"title": item.title, "text": t, "url": item.url})
        return chunks
