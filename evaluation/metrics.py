from typing import List, Dict, Any
from agents.base import AgentSolution
from arena.ranking import LeaderboardEntry


class ArenaMetrics:
    """Computes comparative metrics, tool allocation impact, and evolutionary progression."""

    @staticmethod
    def compute_tool_allocation_experiment(leaderboard: List[LeaderboardEntry]) -> Dict[str, Any]:
        """
        Compare performance between tool-equipped (Tavily) agents and no-tool agents.
        Formula: Improvement = (Score_tavily - Score_notool) / Score_notool
        """
        tavily_entries = [e for e in leaderboard if "tavily" in e.tools]
        notool_entries = [e for e in leaderboard if not e.tools]

        avg_score_tavily = sum(e.score for e in tavily_entries) / len(tavily_entries) if tavily_entries else 0.0
        avg_score_notool = sum(e.score for e in notool_entries) / len(notool_entries) if notool_entries else 0.0

        improvement_pct = 0.0
        if avg_score_notool > 0 and avg_score_tavily > 0:
            improvement_pct = round(((avg_score_tavily - avg_score_notool) / avg_score_notool) * 100, 2)

        avg_latency_tavily = sum(e.latency_sec for e in tavily_entries) / len(tavily_entries) if tavily_entries else 0.0
        avg_latency_notool = sum(e.latency_sec for e in notool_entries) / len(notool_entries) if notool_entries else 0.0

        total_tavily_calls = sum(e.tavily_calls for e in tavily_entries)
        total_tokens_tavily = sum(e.total_tokens for e in tavily_entries)
        total_tokens_notool = sum(e.total_tokens for e in notool_entries)

        return {
            "tavily_agent_count": len(tavily_entries),
            "notool_agent_count": len(notool_entries),
            "avg_score_tavily": round(avg_score_tavily, 2),
            "avg_score_notool": round(avg_score_notool, 2),
            "improvement_pct": improvement_pct,
            "avg_latency_tavily_sec": round(avg_latency_tavily, 2),
            "avg_latency_notool_sec": round(avg_latency_notool, 2),
            "total_tavily_calls": total_tavily_calls,
            "total_tokens_tavily": total_tokens_tavily,
            "total_tokens_notool": total_tokens_notool,
            "hypothesis_validated": avg_score_tavily > avg_score_notool if (tavily_entries and notool_entries) else None
        }

    @staticmethod
    def compute_generation_progress(generation_leaderboards: Dict[int, List[LeaderboardEntry]]) -> Dict[str, Any]:
        """Track how average and top scores evolve across generations."""
        gen_stats = []
        for gen, entries in sorted(generation_leaderboards.items()):
            if not entries:
                continue
            scores = [e.score for e in entries]
            avg_s = sum(scores) / len(scores)
            top_s = max(scores)
            gen_stats.append({
                "generation": gen,
                "top_score": round(top_s, 2),
                "avg_score": round(avg_s, 2),
                "winner": entries[0].agent_name,
                "winner_role": entries[0].role
            })

        overall_gain = 0.0
        if len(gen_stats) >= 2:
            overall_gain = round(gen_stats[-1]["avg_score"] - gen_stats[0]["avg_score"], 2)

        return {
            "generations": gen_stats,
            "overall_gain": overall_gain
        }
