from typing import Dict, Any, List
import groq
from config import GROQ_API_KEY, DEFAULT_MODEL
from agents.base import BaseAgent


TASK_ANALYSIS_PROMPT = """You are the Arena Tool Allocator & Task Analyzer.
Analyze the user's task to determine required tools:

TASK:
\"\"\"{task}\"\"\"

Task Archetypes:
- Current market research (Tavily mandatory)
- Current technology comparison (Tavily mandatory)
- Research literature / recent benchmarks (Tavily mandatory)
- System design (Tavily optional)
- Algorithm problem (Static reasoning - Tavily not needed)
- Mathematical derivation (Static reasoning - Tavily not needed)

Respond in JSON format:
{{
  "category": "one of the archetypes above",
  "requires_live_research": true or false,
  "confidence": <float 0.0-1.0>,
  "reasoning": "brief justification",
  "recommended_tavily_ratio": <float 0.0 to 1.0, e.g. 0.5 for half the population>
}}
"""


class ToolAllocator:
    """Dynamically analyzes tasks and optimizes tool allocations across the agent population."""

    def __init__(self, groq_api_key: str = ""):
        self.api_key = groq_api_key or GROQ_API_KEY

    def analyze_task(self, task: str) -> Dict[str, Any]:
        """Classify task and recommend tool allocation ratio."""
        if not self.api_key:
            return {
                "category": "General",
                "requires_live_research": True,
                "confidence": 0.5,
                "reasoning": "Default heuristic allocation",
                "recommended_tavily_ratio": 0.5
            }

        client = groq.Groq(api_key=self.api_key)
        try:
            res = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise task classifier. Output valid JSON only."},
                    {"role": "user", "content": TASK_ANALYSIS_PROMPT.format(task=task)}
                ],
                temperature=0.1,
                max_tokens=250
            )
            parsed = BaseAgent.extract_json(res.choices[0].message.content or "")
            if parsed and "requires_live_research" in parsed:
                return parsed
        except Exception as e:
            print(f"[ToolAllocator] Notice: Fallback heuristic used: {e}")

        # Heuristic fallback if LLM is unavailable
        time_keywords = ["latest", "recent", "2024", "2025", "2026", "current", "news", "trend", "compare"]
        algo_keywords = ["binary search", "fibonacci", "derive", "sort", "prove", "math", "leetcode"]

        task_lower = task.lower()
        if any(k in task_lower for k in algo_keywords) and not any(k in task_lower for k in time_keywords):
            return {
                "category": "Algorithm / Derivation",
                "requires_live_research": False,
                "confidence": 0.9,
                "reasoning": "Foundational algorithm/math problem not requiring external web data.",
                "recommended_tavily_ratio": 0.0
            }
        else:
            return {
                "category": "Current Technology / Market Research",
                "requires_live_research": True,
                "confidence": 0.85,
                "reasoning": "Task involves temporal comparative or market analysis.",
                "recommended_tavily_ratio": 0.5
            }
