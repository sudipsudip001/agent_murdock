import json

from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric

# 1. Import the Ollama wrapper from deepeval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, ToolCall

# 2. Initialize your local Ollama judge
# (Replace "llama3" with the exact name of the model you have pulled in Ollama)
local_judge = OllamaModel(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.0,  # Keep it at 0 for consistent evaluation scores
)


def run_my_agent(_user_query: str) -> dict[str, str]:
    # Your agent runs here...
    return {"output": "According to the HR handbook...", "tool_used": "vector_search"}


with open("deep_eval.json") as f:
    raw_data = json.load(f)

test_cases = []
for row in raw_data:
    agent_response = run_my_agent(row["input"])
    expected = [ToolCall(name=t["name"]) for t in row["expected_tools"]]
    actual_tools = [ToolCall(name=agent_response["tool_used"])]

    test_case = LLMTestCase(
        input=row["input"],
        actual_output=agent_response["output"],
        tools_called=actual_tools,
        expected_tools=expected,
    )
    test_cases.append(test_case)

# 3. Pass your local judge to the metric
metric = ToolCorrectnessMetric(
    threshold=0.5,
    model=local_judge,  # <-- This tells DeepEval to use Ollama instead of OpenAI
)

evaluate(test_cases, [metric])
