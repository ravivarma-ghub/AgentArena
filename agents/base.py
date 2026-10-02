import time
import json
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

import groq
from config import GROQ_API_KEY, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class AgentConfig(BaseModel):
    id: int
    name: str = ""
    role: str = "General Problem Solver"
    tools: List[str] = Field(default_factory=list)
    strategy: str = "Balanced Tradeoff"
    model: str = DEFAULT_MODEL
    system_prompt: Optional[str] = None
    generation_survived: int = 1


class AgentSolution(BaseModel):
    agent_id: int
    agent_name: str
    role: str
    strategy: str
    generation: int
    task: str
    answer: str
    tools_used: List[str] = Field(default_factory=list)
    tavily_calls: int = 0
    search_queries: List[str] = Field(default_factory=list)
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    score: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class EvaluationResult(BaseModel):
    evaluator_id: str
    evaluator_type: str  # "judge" or "peer"
    target_agent_id: int
    generation: int
    scores: Dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    raw_response: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class BaseAgent:
    """Core base class providing LLM client interaction, token accounting, and safety fallbacks."""

    def __init__(self, config: AgentConfig, groq_api_key: Optional[str] = None):
        self.config = config
        self.agent_id = config.id
        self.name = config.name or f"Agent {self.agent_id:02d}"
        self.role = config.role
        self.tools = config.tools
        self.strategy = config.strategy
        self.model = config.model
        self.score: float = 0.0
        self.history: List[AgentSolution] = []
        self.total_tokens_used: int = 0
        self.total_latency: float = 0.0

        api_key = groq_api_key or GROQ_API_KEY
        if not api_key:
            raise ValueError(f"Groq API key not provided for {self.name}")
        self.client = groq.Groq(api_key=api_key)

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute chat completion with timing, token accounting, and error handling."""
        start_time = time.time()
        target_model = model or self.model

        kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            kwargs["response_format"] = response_format

        retries = 3
        last_error = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - start_time
                content = response.choices[0].message.content or ""

                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0
                total_tokens = usage.total_tokens if usage else 0

                self.total_tokens_used += total_tokens
                self.total_latency += elapsed

                return {
                    "content": content,
                    "latency": elapsed,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "model": target_model
                }
            except Exception as e:
                last_error = e
                # Exponential backoff on rate limits
                time.sleep(1.5 * (attempt + 1))

        # Fallback if retries failed
        elapsed = time.time() - start_time
        return {
            "content": f"[Error: LLM call failed after {retries} attempts: {str(last_error)}]",
            "latency": elapsed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": target_model
        }

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """Extract and parse JSON from model responses reliably."""
        text = text.strip()
        # Direct parse attempt
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try markdown code fences ```json ... ```
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try searching for outermost curly braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return {}
