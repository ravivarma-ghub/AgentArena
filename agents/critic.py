from typing import Any, Dict, List, Optional
from agents.base import BaseAgent, AgentConfig, EvaluationResult, AgentSolution
from prompts.critic import PEER_CRITIC_PROMPT
from config import EVALUATION_CRITERIA


class CriticAgent(BaseAgent):
    """Peer agent that reviews and critiques another agent's solution."""

    def __init__(self, config: AgentConfig, groq_api_key: Optional[str] = None):
        super().__init__(config, groq_api_key)

    def critique(self, task: str, target_solution: AgentSolution, generation: int = 1) -> EvaluationResult:
        """Provide a peer review and score for target agent's solution."""
        prompt = PEER_CRITIC_PROMPT.format(
            critic_name=self.name,
            critic_role=self.role,
            critic_strategy=self.strategy,
            task=task,
            solution=target_solution.answer
        )

        messages = [
            {"role": "system", "content": f"You are peer reviewer {self.name} ({self.role}). Give honest, rigorous critique."},
            {"role": "user", "content": prompt}
        ]

        llm_res = self._call_llm(messages, temperature=0.3, max_tokens=1024)
        parsed = self.extract_json(llm_res["content"])

        try:
            peer_score = round(float(parsed.get("peer_score", 7.5)), 1)
        except (ValueError, TypeError):
            peer_score = 7.5

        strengths = parsed.get("strengths", ["Solid foundational points"])
        weaknesses = parsed.get("weaknesses", ["Needs deeper risk assessment"])
        actionable_tips = parsed.get("actionable_tips", ["Address hardware constraints directly"])

        if not isinstance(strengths, list):
            strengths = [str(strengths)]
        if not isinstance(weaknesses, list):
            weaknesses = [str(weaknesses)]
        if not isinstance(actionable_tips, list):
            actionable_tips = [str(actionable_tips)]

        summary = parsed.get("critique_summary", "")
        if summary:
            weaknesses.insert(0, f"[Summary]: {summary}")

        scores = {crit: peer_score for crit in EVALUATION_CRITERIA}

        return EvaluationResult(
            evaluator_id=self.name,
            evaluator_type="peer",
            target_agent_id=target_solution.agent_id,
            generation=generation,
            scores=scores,
            overall=peer_score,
            strengths=[str(s) for s in strengths if s],
            weaknesses=[str(w) for w in weaknesses if w],
            improvements=[str(a) for a in actionable_tips if a],
            raw_response=llm_res["content"]
        )
