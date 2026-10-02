import json
from typing import Any, Dict, List, Optional
from agents.base import AgentConfig, AgentSolution, EvaluationResult
from agents.solver import SolverAgent
from tools.tavily import TavilyTool
from prompts.solver import RESEARCH_DECISION_PROMPT


class ResearcherAgent(SolverAgent):
    """
    Research agent equipped with Tavily.
    Intelligently analyzes whether web research is necessary before making external queries.
    """

    def __init__(self, config: AgentConfig, groq_api_key: Optional[str] = None, tavily_api_key: Optional[str] = None):
        super().__init__(config, groq_api_key)
        self.tavily_tool = TavilyTool(api_key=tavily_api_key)
        self.tavily_calls_count = 0
        self.search_history: List[Dict[str, Any]] = []

    def decide_research_need(self, task: str) -> Dict[str, Any]:
        """
        Ask LLM whether the given task requires fresh web research.
        Returns a dict: {"needs_research": bool, "reasoning": str, "search_query": str}
        """
        prompt = RESEARCH_DECISION_PROMPT.format(task=task)
        messages = [
            {"role": "system", "content": "You are a precise routing classifier. Respond only in strict JSON."},
            {"role": "user", "content": prompt}
        ]
        res = self._call_llm(messages, temperature=0.1, max_tokens=300)
        parsed = self.extract_json(res["content"])
        
        needs_research = bool(parsed.get("needs_research", False))
        search_query = parsed.get("search_query") or task
        reasoning = parsed.get("reasoning", "Static knowledge sufficient")

        return {
            "needs_research": needs_research,
            "search_query": search_query,
            "reasoning": reasoning
        }

    def _execute_research_if_needed(self, task: str) -> Dict[str, Any]:
        """Execute search only if research is needed and tool is available."""
        if not self.tavily_tool.is_available:
            return {
                "researched": False,
                "context": "",
                "query": "",
                "calls": 0
            }

        decision = self.decide_research_need(task)
        if not decision["needs_research"]:
            return {
                "researched": False,
                "context": "",
                "query": "",
                "reasoning": decision["reasoning"],
                "calls": 0
            }

        # Perform search
        query = decision["search_query"] or task
        search_result = self.tavily_tool.search(query=query, max_results=4)
        self.tavily_calls_count += 1
        self.search_history.append({"query": query, "result": search_result})

        return {
            "researched": True,
            "context": search_result.get("formatted_context", ""),
            "query": query,
            "reasoning": decision["reasoning"],
            "calls": 1
        }

    def solve(self, task: str, generation: int = 1, research_context: str = "") -> AgentSolution:
        research_info = self._execute_research_if_needed(task)
        combined_context = research_context
        if research_info["researched"]:
            combined_context = f"{combined_context}\n{research_info['context']}".strip()

        solution = super().solve(task, generation=generation, research_context=combined_context)

        # Annotate tool metrics
        if research_info["researched"]:
            solution.tools_used = ["tavily"]
            solution.tavily_calls = research_info["calls"]
            solution.search_queries = [research_info["query"]]
        else:
            solution.tools_used = []
            solution.tavily_calls = 0

        return solution

    def improve(
        self,
        task: str,
        previous_solution: AgentSolution,
        evaluations: List[EvaluationResult],
        generation: int,
        research_context: str = ""
    ) -> AgentSolution:
        # Check if weakness indicates missing up-to-date data
        all_weakness_texts = []
        for ev in evaluations:
            ev_w = getattr(ev, "weaknesses", None) or (ev.get("weaknesses", []) if isinstance(ev, dict) else [])
            all_weakness_texts.extend(ev_w)
        needs_more_data = any("source" in w.lower() or "evidence" in w.lower() or "recent" in w.lower()
                              for w in all_weakness_texts)
        
        combined_context = research_context
        additional_calls = 0
        search_queries = list(previous_solution.search_queries)

        if needs_more_data and self.tavily_tool.is_available:
            query = f"{task} benchmarks latest facts"
            search_result = self.tavily_tool.search(query=query, max_results=3)
            self.tavily_calls_count += 1
            additional_calls = 1
            search_queries.append(query)
            combined_context = f"{combined_context}\n[Refined Research]:\n{search_result.get('formatted_context', '')}"

        improved = super().improve(task, previous_solution, evaluations, generation, research_context=combined_context)
        improved.tavily_calls = previous_solution.tavily_calls + additional_calls
        improved.search_queries = search_queries
        if improved.tavily_calls > 0:
            improved.tools_used = ["tavily"]

        return improved
