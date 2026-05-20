from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_token_count: int
    total_token_count: int


class Citation(BaseModel):
    src: str
    page: int


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]


class GenResponse(BaseModel):
    response: str
    citations: dict[str, Citation]
    token_use: TokenUsage
