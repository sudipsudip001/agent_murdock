from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_token_count: int
    total_token_count: int


class Citation(BaseModel):
    src: str = Field(description="The source document name")
    page: int = Field(description="The page number")


class RAGResponse(BaseModel):
    answer: str = Field(description="Answer with inline citations like [1], [2]")
    citations: list[Citation] = Field(description="List of cited sources")


class GenResponse(BaseModel):
    response: str
    citations: dict[str, Citation]
    token_use: TokenUsage


class Context(BaseModel):
    title: str
    url: str
    text: str
