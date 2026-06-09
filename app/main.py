from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException

from app.models.request import QuestionRequest, ResumeRequest
from app.models.response import RunResult
from app.pipeline.agent.graph import Graph

graph = Graph()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await graph.setup()
    yield
    await graph.teardown()


app = FastAPI(lifespan=lifespan)


@app.get("/")  # type: ignore[misc]
def root_message() -> dict[str, str]:
    return {"message": "Welcome to agent_murdock"}


@app.post("/ask")  # type: ignore[misc]
async def ask_question(payload: QuestionRequest) -> RunResult:
    try:
        return await graph.run(
            question=payload.question,
            thread_id=payload.thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/resume")  # type: ignore[misc]
async def resume_question(payload: ResumeRequest) -> RunResult:
    try:
        return await graph.resume(
            thread_id=payload.thread_id,
            decision=payload.decision,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
