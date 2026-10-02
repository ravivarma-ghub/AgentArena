import random
from typing import List, Dict, Union
from agents.base import AgentSolution, EvaluationResult
from agents.critic import CriticAgent
from agents.solver import SolverAgent
from agents.researcher import ResearcherAgent


class PeerTournament:
    """Orchestrates sampled peer critiques among agents without combinatorial explosion."""

    def __init__(self, sample_k: int = 2, groq_api_key: str = ""):
        self.sample_k = sample_k
        self.groq_api_key = groq_api_key

    def conduct_peer_reviews(
        self,
        task: str,
        solutions: List[AgentSolution],
        population: List[Union[SolverAgent, ResearcherAgent]],
        generation: int = 1
    ) -> List[EvaluationResult]:
        """
        Sample k peer critics for each candidate answer.
        Agents do not critique themselves.
        """
        evaluations: List[EvaluationResult] = []
        if len(population) < 2:
            return evaluations

        agent_dict = {a.agent_id: a for a in population}

        for sol in solutions:
            # Candidates who can critique this agent (all except self)
            candidate_critics = [a for a in population if a.agent_id != sol.agent_id]
            k = min(self.sample_k, len(candidate_critics))
            selected_critics = random.sample(candidate_critics, k)

            for critic_agent in selected_critics:
                critic = CriticAgent(
                    config=critic_agent.config,
                    groq_api_key=self.groq_api_key
                )
                try:
                    review = critic.critique(task=task, target_solution=sol, generation=generation)
                    evaluations.append(review)
                except Exception as e:
                    print(f"[PeerTournament] Review error by {critic_agent.name} on {sol.agent_name}: {e}")

        return evaluations
