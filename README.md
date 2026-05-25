# agent_murdock

`agent_murdock` is an agentic RAG implementation for legal workflow.

### Tools used:
- `weaviate` as vector database
- `gemini` as agentic LLM
- `langgraph` as agentic framework
- `serper.dev` as web search tool
- `nomic-ai/nomic-embed-text-v1` as the embedder model


### Git workflow to follow
```mermaid
gitGraph
    commit id: "Initial"
    branch develop
    checkout develop
    branch vector_db
    branch web_search
    checkout vector_db
    commit id: "Use vector database"
    checkout web_search
    commit id: "Add web search"
    checkout vector_db
    commit id: "Auth Logic"
    checkout develop
    merge vector_db id: "DB merge"
    merge web_search id: "Web search integrate"
    checkout main
    merge develop id: "Release Feature"
```
