from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


class Chunker:
    MARKDOWN_SEPARATORS = ["\n#{1,6} ", "```\n", "\n\n", "\n", " ", ""]

    def __init__(
        self,
        doc_list: list[str],
        chunk_size: int,
        chunk_overlap: int,
        embedder_model_name: str = "nomic-ai/nomic-embed-text-v1",
    ) -> None:
        self.PDF_LIST = doc_list
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = chunk_overlap
        self.EMBEDDING_MODEL_NAME = embedder_model_name
        self._raw_knowledge_base: list[Document] | None = None

    @property
    def raw_knowledge_base(self) -> list[Document]:
        if self._raw_knowledge_base is None:
            self._raw_knowledge_base = []
            for pdf_path in self.PDF_LIST:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                for page in pages:
                    self._raw_knowledge_base.append(
                        Document(
                            page_content=page.page_content,
                            metadata={
                                "source": pdf_path,
                                "page": page.metadata["page"] + 1,
                            },
                        )
                    )
        return self._raw_knowledge_base

    def chunk_docs(
        self,
    ) -> list[Document]:
        all_docs = self.raw_knowledge_base
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME),
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )
        seen, chunks = set(), []
        for doc in text_splitter.split_documents(all_docs):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                chunks.append(doc)
        return chunks
