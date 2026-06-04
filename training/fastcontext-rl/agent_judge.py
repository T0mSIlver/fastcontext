import json
import os
import pathlib
import re

from openai import AsyncOpenAI

system_prompt = pathlib.Path("agent_judge.md").read_text(encoding="utf-8").strip()


async def call_llm(base_url: str, api_key: str, model: str, messages: list) -> str:

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


def create_messages(query: str, ground_truth: str, final_answer: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{query}\n\n<ground_truth>\n{ground_truth}\n</ground_truth>\n\n<final_answer>\n{final_answer}\n</final_answer>",
        },
    ]
    return messages


def parse_judge_result(text: str) -> int:
    if "```json" in text:
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        result_json = json.loads(text)
        score = result_json.get("score", 0)
        score = int(score) * 0.1
        return score
    except Exception as e:
        print(f"Error parsing judge result: {e}")
        return 0


class JudgeAgent:
    def __init__(self, base_url=None, api_key=None, model=None):
        base_url = os.environ.get("BASE_URL", base_url)
        api_key = os.environ.get("API_KEY", api_key)
        model = os.environ.get("MODEL", model)

        assert base_url is not None, "BASE_URL must be set"
        assert api_key is not None, "API_KEY must be set"
        assert model is not None, "MODEL must be set"

        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    async def score(self, query: str, label: str, pred: str) -> int:
        try:
            messages = create_messages(query, label, pred)
            result = await call_llm(base_url=self.base_url, api_key=self.api_key, model=self.model, messages=messages)
            return parse_judge_result(result)
        except Exception as e:
            print(f"Error of call llm as judge: {e}")
        return 0

    async def judge(self, query: str, citation_contents: str) -> int:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{query}\n\n<final_answer>\n{citation_contents}\n</final_answer>",
                },
            ]
            result = await call_llm(base_url=self.base_url, api_key=self.api_key, model=self.model, messages=messages)
            return parse_judge_result(result)
        except Exception as e:
            print(f"Error of call llm as judge: {e}")
        return 0


if __name__ == "__main__":
    import asyncio
    from utils import read_citations_content

    agent = JudgeAgent()

    async def main():
        c = read_citations_content(
            [
                {
                    "path": "/testbed/azure-functions-durable-js/src/classes.ts",
                    "line_range": "50-58",
                    "start_line": 50,
                    "end_line": 58,
                },
                {
                    "path": "/testbed/azure-functions-durable-js/src/durableorchestrationclient.ts",
                    "line_range": "16-23",
                    "start_line": 16,
                    "end_line": 23,
                },
            ],
            workspace="/testbed/",
        )
        s = await agent.judge(
            query="Explore this codebase and find all relevant files related to:\n1. `IHttpRequest` and `IHttpResponse` interfaces/classes",
            citation_contents=c,
        )
        print(f"Score: {s}")

    asyncio.run(main())
