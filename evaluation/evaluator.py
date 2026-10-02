from typing import List, Dict, Any, Optional
from agents.base import AgentSolution, EvaluationResult
from agents.judge import JudgeAgent
from prompts.judge import JUDGE_SYSTEM_PROMPTS


class MultiJudgeEvaluator:
    """
    Coordinates a multi-judge panel with distinct personas to evaluate candidate solutions,
    averaging scores to mitigate individual judge bias (Phase 6 & 7).
    """

    def __init__(self, num_judges: int = 3, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key
        persona_names = list(JUDGE_SYSTEM_PROMPTS.keys())
        self.judges: List[JudgeAgent] = []

        count = min(num_judges, len(persona_names))
        for i in range(count):
            persona = persona_names[i]
            judge = JudgeAgent(
                judge_id=f"Judge {i+1} ({persona})",
                persona_name=persona,
                groq_api_key=self.groq_api_key
            )
            self.judges.append(judge)

    def evaluate_solutions(
        self,
        task: str,
        solutions: List[AgentSolution],
        generation: int = 1
    ) -> List[EvaluationResult]:
        """Run all judges across all candidate solutions."""
        all_evaluations: List[EvaluationResult] = []

        for sol in solutions:
            for judge in self.judges:
                try:
                    result = judge.evaluate(task=task, solution=sol, generation=generation)
                    all_evaluations.append(result)
                except Exception as e:
                    print(f"[MultiJudgeEvaluator] Evaluation error by {judge.name} on Agent {sol.agent_id}: {e}")

        return all_evaluations
