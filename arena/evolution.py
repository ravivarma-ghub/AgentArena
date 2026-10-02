import random
from typing import List, Union, Dict, Any
from agents.base import AgentConfig
from agents.solver import SolverAgent
from agents.researcher import ResearcherAgent
from arena.ranking import LeaderboardEntry


MUTATION_STRATEGIES = [
    "Evidence-first",
    "Pragmatic Implementation",
    "Stress-test & Risk-analysis",
    "First-principles",
    "Startup-lean",
    "Balanced Tradeoff"
]


class EvolutionEngine:
    """Manages natural selection, strategy mutation, and population adaptation across generations."""

    def __init__(self, survival_rate: float = 0.75):
        self.survival_rate = survival_rate

    def evolve_population(
        self,
        current_population: List[Union[SolverAgent, ResearcherAgent]],
        leaderboard: List[LeaderboardEntry],
        groq_api_key: str = "",
        tavily_api_key: str = ""
    ) -> Dict[str, Any]:
        """
        Select survivors and mutate bottom-tier agents into a new generation population.
        """
        if len(current_population) <= 2:
            # For small test populations (<=2), mutate strategy of the lowest scorer
            return {
                "survivor_ids": [a.agent_id for a in current_population],
                "mutated_ids": [],
                "population": current_population,
                "mutations_applied": []
            }

        cutoff_index = max(1, int(len(leaderboard) * self.survival_rate))
        survivor_entries = leaderboard[:cutoff_index]
        bottom_entries = leaderboard[cutoff_index:]

        survivor_ids = {e.agent_id for e in survivor_entries}
        top_strategies = [e.strategy for e in survivor_entries]

        # Check if Tavily agents in top tier outperformed
        tavily_survivors = sum(1 for e in survivor_entries if "tavily" in e.tools)
        favor_tavily = tavily_survivors > (len(survivor_entries) / 2)

        agent_by_id = {a.agent_id: a for a in current_population}
        new_population: List[Union[SolverAgent, ResearcherAgent]] = []
        mutations_applied = []

        # Retain survivors
        for a_id in survivor_ids:
            if a_id in agent_by_id:
                new_population.append(agent_by_id[a_id])

        # Mutate bottom agents
        for entry in bottom_entries:
            old_agent = agent_by_id.get(entry.agent_id)
            if not old_agent:
                continue

            # Pick a new strategy: either adopt from a top survivor or sample mutation pool
            available_strategies = [s for s in MUTATION_STRATEGIES if s != entry.strategy]
            new_strat = random.choice(top_strategies) if (top_strategies and random.random() < 0.6) else random.choice(available_strategies)

            # Determine tools: if tavily is favored by top survivors, consider equipping tavily
            new_tools = list(old_agent.tools)
            if favor_tavily and "tavily" not in new_tools and random.random() < 0.7:
                new_tools.append("tavily")

            mutated_config = AgentConfig(
                id=old_agent.agent_id,
                name=old_agent.name,
                role=old_agent.role,
                tools=new_tools,
                strategy=new_strat,
                generation_survived=old_agent.config.generation_survived + 1
            )

            if "tavily" in new_tools:
                mutated_agent = ResearcherAgent(
                    config=mutated_config,
                    groq_api_key=groq_api_key,
                    tavily_api_key=tavily_api_key
                )
            else:
                mutated_agent = SolverAgent(
                    config=mutated_config,
                    groq_api_key=groq_api_key
                )

            # Preserve history
            mutated_agent.history = list(old_agent.history)
            new_population.append(mutated_agent)

            mutations_applied.append({
                "agent_id": old_agent.agent_id,
                "agent_name": old_agent.name,
                "old_strategy": entry.strategy,
                "new_strategy": new_strat,
                "tools": new_tools,
                "reason": f"Rank {entry.rank} (score {entry.score:.1f}). Mutated strategy towards top performers."
            })

        # Ensure order matches agent IDs
        new_population.sort(key=lambda a: a.agent_id)

        return {
            "survivor_ids": list(survivor_ids),
            "mutated_ids": [e.agent_id for e in bottom_entries],
            "population": new_population,
            "mutations_applied": mutations_applied
        }
