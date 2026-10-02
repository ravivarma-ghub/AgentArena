from typing import Any, Dict, List, Optional
from agents.base import BaseAgent, AgentConfig, EvaluationResult, AgentSolution
from prompts.judge import JUDGE_SYSTEM_PROMPTS, JUDGE_EVALUATION_PROMPT
from config import EVALUATION_CRITERIA


class JudgeAgent(BaseAgent):
    """
    Independent Judge Agent capable of objective multi-criteria grading.
    Uses dedicated judge personas to diversify perspectives.
    """

    def __init__(
        self,
        judge_id: str = "Judge 1",
        persona_name: str = "Balanced Technical Lead",
        groq_api_key: Optional[str] = None
    ):
        config = AgentConfig(
            id=900,
            name=judge_id,
            role="Judge",
            strategy="Objective Multi-Criteria Grading"
        )
        super().__init__(config, groq_api_key)
        self.judge_id = judge_id
        self.persona_name = persona_name
        self.persona_desc = JUDGE_SYSTEM_PROMPTS.get(
            persona_name,
            "You are an impartial, highly rigorous evaluator."
        )

    def evaluate(self, task: str, solution: AgentSolution, generation: int = 1) -> EvaluationResult:
        """Evaluate a single solution across all 6 criteria."""
        user_prompt = JUDGE_EVALUATION_PROMPT.format(
            judge_persona=f"{self.persona_name} - {self.persona_desc}",
            task=task,
            solution=solution.answer
        )

        messages = [
            {"role": "system", "content": self.persona_desc},
            {"role": "user", "content": user_prompt}
        ]

        llm_res = self._call_llm(messages, temperature=0.2, max_tokens=1024)
        parsed = self.extract_json(llm_res["content"])

        scores: Dict[str, float] = {}
        for crit in EVALUATION_CRITERIA:
            val = parsed.get(crit)
            try:
                scores[crit] = round(float(val), 1) if val is not None else 7.0
            except (ValueError, TypeError):
                scores[crit] = 7.0

        # Calculate overall score
        overall_val = parsed.get("overall")
        try:
            overall = round(float(overall_val), 1) if overall_val is not None else round(sum(scores.values()) / len(scores), 1)
        except (ValueError, TypeError):
            overall = round(sum(scores.values()) / len(scores), 1) if scores else 7.0

        strengths = parsed.get("strengths", ["Clear explanation of core concepts"])
        weaknesses = parsed.get("weaknesses", ["Could provide more empirical evidence and quantitative metrics"])
        improvements = parsed.get("improvements", ["Include specific benchmarks and cost calculations"])

        if not isinstance(strengths, list):
            strengths = [str(strengths)]
        if not isinstance(weaknesses, list):
            weaknesses = [str(weaknesses)]
        if not isinstance(improvements, list):
            improvements = [str(improvements)]

        return EvaluationResult(
            evaluator_id=self.judge_id,
            evaluator_type="judge",
            target_agent_id=solution.agent_id,
            generation=generation,
            scores=scores,
            overall=overall,
            strengths=[str(s) for s in strengths if s],
            weaknesses=[str(w) for w in weaknesses if w],
            improvements=[str(i) for i in improvements if i],
            raw_response=llm_res["content"]
        )
