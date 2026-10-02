from typing import Any, Dict, List, Optional
from agents.base import BaseAgent, AgentConfig, AgentSolution, EvaluationResult
from prompts.solver import (
    SOLVER_SYSTEM_PROMPTS,
    STRATEGY_INSTRUCTIONS,
    SOLVE_PROMPT_TEMPLATE,
    IMPROVEMENT_PROMPT_TEMPLATE
)


class SolverAgent(BaseAgent):
    """Standard problem solver agent capable of initial solution and iterative improvement."""

    def __init__(self, config: AgentConfig, groq_api_key: Optional[str] = None):
        super().__init__(config, groq_api_key)
        self.system_prompt = config.system_prompt or SOLVER_SYSTEM_PROMPTS.get(
            self.role,
            "You are an expert analytical problem solver delivering high-quality answers."
        )

    def _get_strategy_detail(self) -> str:
        return STRATEGY_INSTRUCTIONS.get(self.strategy, f"Focus on {self.strategy}.")

    def solve(self, task: str, generation: int = 1, research_context: str = "") -> AgentSolution:
        """Independently solve the task for this generation."""
        research_str = f"RESEARCH & FACTUAL CONTEXT:\n{research_context}\n" if research_context else ""

        user_content = SOLVE_PROMPT_TEMPLATE.format(
            name=self.name,
            role=self.role,
            strategy=self._get_strategy_detail(),
            task=task,
            research_context=research_str
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        llm_res = self._call_llm(messages)

        solution = AgentSolution(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            strategy=self.strategy,
            generation=generation,
            task=task,
            answer=llm_res["content"],
            tools_used=list(self.tools),
            latency_sec=llm_res["latency"],
            prompt_tokens=llm_res["prompt_tokens"],
            completion_tokens=llm_res["completion_tokens"],
            total_tokens=llm_res["total_tokens"],
            score=0.0
        )
        self.history.append(solution)
        return solution

    def improve(
        self,
        task: str,
        previous_solution: AgentSolution,
        evaluations: List[EvaluationResult],
        generation: int,
        research_context: str = ""
    ) -> AgentSolution:
        """Improve previous solution using judge scores, weaknesses, and peer recommendations."""
        # Collate strengths, weaknesses, improvements, and critique notes
        all_strengths = []
        all_weaknesses = []
        all_improvements = []
        critique_notes = []
        scores = []

        for ev in evaluations:
            overall = getattr(ev, "overall", None) or (ev.get("overall", 7.0) if isinstance(ev, dict) else 7.0)
            scores.append(overall)
            ev_strengths = getattr(ev, "strengths", None) or (ev.get("strengths", []) if isinstance(ev, dict) else [])
            all_strengths.extend(ev_strengths)
            ev_weaknesses = getattr(ev, "weaknesses", None) or (ev.get("weaknesses", []) if isinstance(ev, dict) else [])
            all_weaknesses.extend(ev_weaknesses)
            ev_improvements = getattr(ev, "improvements", None) or (ev.get("improvements", []) if isinstance(ev, dict) else [])
            all_improvements.extend(ev_improvements)
            ev_type = getattr(ev, "evaluator_type", None) or (ev.get("evaluator_type", "") if isinstance(ev, dict) else "")
            ev_id = getattr(ev, "evaluator_id", None) or (ev.get("evaluator_id", "Reviewer") if isinstance(ev, dict) else "Reviewer")
            if ev_type == "peer":
                critique_notes.append(f"- Peer {ev_id}: {', '.join(ev_weaknesses[:2])}")

        avg_score = round(sum(scores) / len(scores), 2) if scores else 7.0

        strengths_fmt = "\n".join([f"- {s}" for s in list(dict.fromkeys(all_strengths))[:5]]) or "- Solid baseline reasoning"
        weaknesses_fmt = "\n".join([f"- {w}" for w in list(dict.fromkeys(all_weaknesses))[:5]]) or "- Could add more concrete technical depth"
        improvements_fmt = "\n".join([f"- {i}" for i in list(dict.fromkeys(all_improvements))[:5]]) or "- Add explicit examples and quantify tradeoffs"
        critiques_fmt = "\n".join(critique_notes[:4]) if critique_notes else "- No specific peer critiques"

        research_str = f"UPDATED RESEARCH CONTEXT:\n{research_context}\n" if research_context else ""

        user_content = IMPROVEMENT_PROMPT_TEMPLATE.format(
            name=self.name,
            role=self.role,
            strategy=self._get_strategy_detail(),
            task=task,
            previous_answer=previous_solution.answer,
            score=avg_score,
            strengths=strengths_fmt,
            weaknesses=weaknesses_fmt,
            improvements=improvements_fmt,
            critique_notes=critiques_fmt,
            research_context=research_str,
            generation=generation
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        llm_res = self._call_llm(messages)

        improved_solution = AgentSolution(
            agent_id=self.agent_id,
            agent_name=self.name,
            role=self.role,
            strategy=self.strategy,
            generation=generation,
            task=task,
            answer=llm_res["content"],
            tools_used=list(self.tools),
            latency_sec=llm_res["latency"],
            prompt_tokens=llm_res["prompt_tokens"],
            completion_tokens=llm_res["completion_tokens"],
            total_tokens=llm_res["total_tokens"],
            score=0.0
        )
        self.history.append(improved_solution)
        return improved_solution
