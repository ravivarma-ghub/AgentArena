from typing import Any, Dict, List, Optional
from agents.base import BaseAgent, AgentConfig, AgentSolution, EvaluationResult
from prompts.arbiter import FINAL_ARBITER_SYSTEM_PROMPT, FINAL_ARBITER_SYNTHESIS_PROMPT


class ArbiterAgent(BaseAgent):
    """
    Supreme Final Arbiter Agent.
    Synthesizes the best aspects of top candidate solutions across generations
    and eliminates remaining flaws to forge a superior master solution.
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        config = AgentConfig(
            id=999,
            name="Grand Arbiter",
            role="Supreme Arbiter & Solution Synthesizer",
            strategy="Master Synthesis"
        )
        super().__init__(config, groq_api_key)

    def synthesize(
        self,
        task: str,
        top_solutions: List[AgentSolution],
        evaluations_summary: List[str],
        total_generations: int,
        total_agents: int
    ) -> Dict[str, Any]:
        """Synthesize top solutions into championship answer."""
        candidates_formatted_list = []
        for idx, sol in enumerate(top_solutions, 1):
            candidates_formatted_list.append(
                f"### Candidate {idx}: {sol.agent_name} ({sol.role}) - Score: {sol.score:.1f}\n"
                f"Tools Used: {', '.join(sol.tools_used) or 'None'} | Strategy: {sol.strategy}\n\n"
                f"{sol.answer}\n"
                f"{'-'*40}"
            )

        candidates_formatted = "\n\n".join(candidates_formatted_list)
        critiques_str = "\n".join([f"- {c}" for c in evaluations_summary[:8]]) or "No major outstanding gaps."

        prompt = FINAL_ARBITER_SYNTHESIS_PROMPT.format(
            task=task,
            candidates_formatted=candidates_formatted,
            total_generations=total_generations,
            total_agents=total_agents,
            critique_synthesis=critiques_str
        )

        messages = [
            {"role": "system", "content": FINAL_ARBITER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        llm_res = self._call_llm(messages, temperature=0.4, max_tokens=3000)

        return {
            "master_solution": llm_res["content"],
            "top_candidates": [
                {
                    "agent_id": s.agent_id,
                    "agent_name": s.agent_name,
                    "role": s.role,
                    "score": s.score,
                    "strategy": s.strategy,
                    "tools_used": s.tools_used
                }
                for s in top_solutions
            ],
            "latency": llm_res["latency"],
            "total_tokens": llm_res["total_tokens"]
        }
