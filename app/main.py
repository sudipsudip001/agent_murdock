import uvicorn
from fastapi import FastAPI

from app.dependencies import graph
from app.models.request import QuestionRequest

app = FastAPI()


@app.get("/")  # type: ignore[misc]
def root_message() -> dict[str, str]:
    return {"message": "Welcome to agent_murdock"}


@app.post("/ask")  # type: ignore[misc]
async def ask_question(question: QuestionRequest) -> dict[str, str]:
    answer = await graph.run(question=question.question)
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
