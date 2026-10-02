from typing import Dict, List, Any
from agents.base import AgentSolution, EvaluationResult


class LeaderboardEntry:
    def __init__(
        self,
        rank: int,
        agent_id: int,
        agent_name: str,
        role: str,
        strategy: str,
        tools: List[str],
        score: float,
        dimension_scores: Dict[str, float],
        tavily_calls: int,
        latency_sec: float,
        total_tokens: int,
        strengths: List[str],
        weaknesses: List[str]
    ):
        self.rank = rank
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.role = role
        self.strategy = strategy
        self.tools = tools
        self.score = score
        self.dimension_scores = dimension_scores
        self.tavily_calls = tavily_calls
        self.latency_sec = latency_sec
        self.total_tokens = total_tokens
        self.strengths = strengths
        self.weaknesses = weaknesses

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "role": self.role,
            "strategy": self.strategy,
            "tools": self.tools,
            "score": round(self.score, 2),
            "dimension_scores": {k: round(v, 2) for k, v in self.dimension_scores.items()},
            "tavily_calls": self.tavily_calls,
            "latency_sec": round(self.latency_sec, 2),
            "total_tokens": self.total_tokens,
            "strengths": self.strengths[:3],
            "weaknesses": self.weaknesses[:3]
        }


class Ranker:
    """Computes generational leaderboard by aggregating multi-judge and peer evaluations."""

    @staticmethod
    def calculate_leaderboard(
        solutions: List[AgentSolution],
        evaluations: List[EvaluationResult]
    ) -> List[LeaderboardEntry]:
        """Aggregate evaluations and rank agents by overall score."""
        agent_eval_map: Dict[int, List[EvaluationResult]] = {}
        for ev in evaluations:
            agent_eval_map.setdefault(ev.target_agent_id, []).append(ev)

        entries: List[LeaderboardEntry] = []

        for sol in solutions:
            evals = agent_eval_map.get(sol.agent_id, [])
            if not evals:
                overall_score = 7.0
                dim_scores = {}
                strengths = []
                weaknesses = []
            else:
                overall_score = sum(e.overall for e in evals) / len(evals)
                
                # Aggregate criteria scores
                all_keys = set().union(*[e.scores.keys() for e in evals])
                dim_scores = {}
                for k in all_keys:
                    k_vals = [e.scores[k] for e in evals if k in e.scores]
                    dim_scores[k] = sum(k_vals) / len(k_vals) if k_vals else 7.0

                strengths = []
                weaknesses = []
                for e in evals:
                    strengths.extend(e.strengths)
                    weaknesses.extend(e.weaknesses)

            sol.score = overall_score

            entries.append(LeaderboardEntry(
                rank=0,
                agent_id=sol.agent_id,
                agent_name=sol.agent_name,
                role=sol.role,
                strategy=sol.strategy,
                tools=sol.tools_used,
                score=overall_score,
                dimension_scores=dim_scores,
                tavily_calls=sol.tavily_calls,
                latency_sec=sol.latency_sec,
                total_tokens=sol.total_tokens,
                strengths=list(dict.fromkeys(strengths)),
                weaknesses=list(dict.fromkeys(weaknesses))
            ))

        # Sort descending by score
        entries.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks
        for idx, entry in enumerate(entries, 1):
            entry.rank = idx

        return entries
