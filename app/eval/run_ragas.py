import json
import logging
import os
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)


class Ragged:
    def __init__(
        self,
        dataset: list[dict[str, Any]],
        embedder: str = "thenlper/gte-small",
    ) -> None:
        self.EMBEDDING_MODEL_NAME = embedder
        self._ragas_llm = None
        self._embedder = None

        # ── Validate rows ──────────────────────────────────────────────────────
        required_keys = ["user_input", "retrieved_contexts", "response", "reference"]
        for i, row in enumerate(dataset):
            for key in required_keys:
                if key not in row:
                    logger.warning(f"Row {i} missing key: '{key}'")
                elif row[key] is None:
                    logger.warning(f"Row {i} has None for: '{key}'")
            contexts = row.get("retrieved_contexts")
            if isinstance(contexts, list) and len(contexts) == 0:
                logger.warning(f"Row {i} has empty retrieved_contexts list")
            if isinstance(contexts, str):
                logger.warning(
                    f"Row {i}: retrieved_contexts is a string, should be List[str]"
                )

        # ── Build HuggingFace Dataset (keys already match RAGAS expectations) ──
        self.dataset = Dataset.from_list(
            [
                {
                    "user_input": row["user_input"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "response": row["response"],
                    "reference": row["reference"],
                }
                for row in dataset
            ]
        )
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    # ── Alternative constructor: build directly from agentic RAG turns ────────
    @classmethod
    def from_agent_turns(
        cls,
        turns: list[dict[str, Any]],
        embedder: str = "thenlper/gte-small",
    ) -> "Ragged":
        """
        Accepts a list of raw agent turns:
        {
            "user_input":  str,           # original user question
            "agent_response": str,        # raw AIMessage.content (JSON string)
            "reference":   str,           # ground-truth answer
        }
        Parses the agent_response JSON and extracts retrieved_contexts + response.
        """
        rows: list[dict[str, Any]] = []
        for i, turn in enumerate(turns):
            try:
                parsed = json.loads(turn["agent_response"])
                rows.append(
                    {
                        "user_input": turn["user_input"],
                        "retrieved_contexts": parsed.get("retrieved_contexts", []),
                        "response": parsed.get("answer", ""),
                        "reference": turn["reference"],
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Turn {i} could not be parsed, skipping: {e}")
        return cls(rows, embedder=embedder)

    # ── Lazy properties ────────────────────────────────────────────────────────
    @property
    def ragas_llm(self) -> LangchainLLMWrapper:
        if self._ragas_llm is None:
            self._ragas_llm = LangchainLLMWrapper(
                ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    api_key=os.getenv("GROQ_API_KEY"),
                )
            )
        return self._ragas_llm

    @property
    def embedder(self) -> LangchainEmbeddingsWrapper:
        if self._embedder is None:
            self._embedder = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL_NAME)
            )
        return self._embedder

    # ── Evaluate ───────────────────────────────────────────────────────────────
    def score(self) -> pd.DataFrame:
        results = evaluate(
            self.dataset,
            metrics=self.metrics,
            llm=self.ragas_llm,
            embeddings=self.embedder,
        )
        return results.to_pandas()
