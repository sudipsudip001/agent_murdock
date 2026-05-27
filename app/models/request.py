from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str
    thread_id: str = Field(description="Unique ID for user session")
