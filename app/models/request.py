from typing import Literal

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str
    thread_id: str = Field(description="Unique ID for user session")


class ResumeRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
