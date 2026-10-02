import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Union

from config import (
    GROQ_API_KEY,
    TAVILY_API_KEY,
    DEFAULT_NUM_GENERATIONS,
    DEFAULT_POPULATION_SIZE
)
from agents.base import AgentSolution, EvaluationResult
from agents.solver import SolverAgent
from agents.researcher import ResearcherAgent
from agents.arbiter import ArbiterAgent
from arena.population import create_population
from arena.tool_allocator import ToolAllocator
from arena.ranking import Ranker, LeaderboardEntry
from arena.evolution import EvolutionEngine
from arena.tournament import PeerTournament
from evaluation.evaluator import MultiJudgeEvaluator
from evaluation.metrics import ArenaMetrics
from memory.store import ArenaMemoryStore


class ArenaController:
    """
    Master tournament controller for the Agent Arena.
    Orchestrates the entire evolutionary competitive lifecycle:
    Independent Solving -> Multi-Judge Evaluation -> Peer Critiques -> Leaderboards
    -> Selection & Mutation -> Iterative Improvement -> Final Arbiter Synthesis.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        num_judges: int = 3,
        survival_rate: float = 0.75,
        peer_sample_k: int = 2
    ):
        self.groq_api_key = groq_api_key or GROQ_API_KEY
        self.tavily_api_key = tavily_api_key or TAVILY_API_KEY
        self.num_judges = num_judges
        self.survival_rate = survival_rate
        self.peer_sample_k = peer_sample_k

        self.evaluator = MultiJudgeEvaluator(num_judges=self.num_judges, groq_api_key=self.groq_api_key)
        self.tool_allocator = ToolAllocator(groq_api_key=self.groq_api_key)
        self.evolution_engine = EvolutionEngine(survival_rate=self.survival_rate)
        self.peer_tournament = PeerTournament(sample_k=self.peer_sample_k, groq_api_key=self.groq_api_key)
        self.arbiter = ArbiterAgent(groq_api_key=self.groq_api_key)
        self.memory_store = ArenaMemoryStore()

    def run_tournament(
        self,
        task: str,
        num_agents: int = DEFAULT_POPULATION_SIZE,
        num_generations: int = DEFAULT_NUM_GENERATIONS,
        enable_peer_evaluation: bool = True,
        enable_dynamic_tool_assignment: bool = True,
        progress_callback: Optional[Callable[[str, float, Optional[Dict[str, Any]]], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute full competitive multi-agent tournament.
        """
        run_id = f"arena_run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        start_tournament_time = time.time()

        def notify(msg: str, progress: float = 0.0, data: Optional[Dict[str, Any]] = None):
            if progress_callback:
                progress_callback(msg, progress, data)

        notify("Initializing Agent Arena tournament...", 0.02)

        # -------------------------------------------------------------
        # Phase 14: Dynamic Tool Assignment & Task Analysis
        # -------------------------------------------------------------
        task_analysis = {}
        if enable_dynamic_tool_assignment:
            notify("Analyzing task requirements and tool needs...", 0.05)
            task_analysis = self.tool_allocator.analyze_task(task)

        # -------------------------------------------------------------
        # Phase 4: Create Agent Population
        # -------------------------------------------------------------
        notify(f"Spawning population of {num_agents} diverse agents...", 0.10)
        population = create_population(
            population_size=num_agents,
            groq_api_key=self.groq_api_key,
            tavily_api_key=self.tavily_api_key
        )

        generation_history: Dict[int, Dict[str, Any]] = {}
        generation_leaderboards: Dict[int, List[LeaderboardEntry]] = {}
        current_solutions: Dict[int, AgentSolution] = {}

        total_steps = num_generations * 4 + 2
        step_counter = 0

        for gen in range(1, num_generations + 1):
            gen_start_time = time.time()
            notify(f"Starting Generation {gen}/{num_generations}...", 0.10 + (gen - 1) * 0.35)

            # ---------------------------------------------------------
            # Step 1: Solving Phase (Phase 5 / Phase 10)
            # ---------------------------------------------------------
            solutions_this_gen: List[AgentSolution] = []

            for idx, agent in enumerate(population, 1):
                notify(
                    f"[Gen {gen}] {agent.name} ({agent.role}) solving independently...",
                    min(0.90, 0.12 + ((gen - 1) * 0.35) + (idx / len(population)) * 0.10)
                )

                if gen == 1:
                    # Round 1: Independent solving without seeing peers
                    sol = agent.solve(task, generation=1)
                else:
                    # Evolutionary Round: Improve with feedback from previous generation
                    prev_sol = current_solutions.get(agent.agent_id)
                    prev_evals = generation_history[gen - 1]["evaluations"].get(agent.agent_id, [])
                    if prev_sol:
                        sol = agent.improve(
                            task=task,
                            previous_solution=prev_sol,
                            evaluations=prev_evals,
                            generation=gen
                        )
                    else:
                        sol = agent.solve(task, generation=gen)

                solutions_this_gen.append(sol)
                current_solutions[agent.agent_id] = sol

            # ---------------------------------------------------------
            # Step 2: Multi-Judge Panel Evaluation (Phase 6 & 7)
            # ---------------------------------------------------------
            notify(f"[Gen {gen}] Multi-judge panel evaluating {len(solutions_this_gen)} solutions...", 0.20 + (gen - 1) * 0.35)
            judge_evaluations = self.evaluator.evaluate_solutions(
                task=task,
                solutions=solutions_this_gen,
                generation=gen
            )

            # ---------------------------------------------------------
            # Step 3: Peer Critiques (Phase 8)
            # ---------------------------------------------------------
            peer_evaluations: List[EvaluationResult] = []
            if enable_peer_evaluation and len(population) > 1:
                notify(f"[Gen {gen}] Conducting peer critiques...", 0.25 + (gen - 1) * 0.35)
                peer_evaluations = self.peer_tournament.conduct_peer_reviews(
                    task=task,
                    solutions=solutions_this_gen,
                    population=population,
                    generation=gen
                )

            all_evaluations = judge_evaluations + peer_evaluations

            # ---------------------------------------------------------
            # Step 4: Ranking & Leaderboard (Phase 9)
            # ---------------------------------------------------------
            notify(f"[Gen {gen}] Aggregating scores and computing leaderboard...", 0.30 + (gen - 1) * 0.35)
            leaderboard = Ranker.calculate_leaderboard(solutions_this_gen, all_evaluations)
            generation_leaderboards[gen] = leaderboard

            # Map evaluations by target agent
            agent_evals_map: Dict[int, List[EvaluationResult]] = {}
            for ev in all_evaluations:
                agent_evals_map.setdefault(ev.target_agent_id, []).append(ev)

            # Tool experiment for this generation
            tool_experiment_gen = ArenaMetrics.compute_tool_allocation_experiment(leaderboard)

            generation_history[gen] = {
                "generation": gen,
                "duration_sec": round(time.time() - gen_start_time, 2),
                "leaderboard": [e.to_dict() for e in leaderboard],
                "solutions": [s.model_dump() for s in solutions_this_gen],
                "evaluations": {k: [e.model_dump() for e in v] for k, v in agent_evals_map.items()},
                "tool_experiment": tool_experiment_gen
            }

            # ---------------------------------------------------------
            # Step 5: Survival & Mutation (Phase 12)
            # ---------------------------------------------------------
            if gen < num_generations:
                notify(f"[Gen {gen}] Applying natural selection and strategy mutation...", 0.35 + (gen - 1) * 0.35)
                evolution_result = self.evolution_engine.evolve_population(
                    current_population=population,
                    leaderboard=leaderboard,
                    groq_api_key=self.groq_api_key,
                    tavily_api_key=self.tavily_api_key
                )
                population = evolution_result["population"]
                generation_history[gen]["evolution_actions"] = evolution_result["mutations_applied"]

        # -------------------------------------------------------------
        # Phase 15: Grand Final Arbiter
        # -------------------------------------------------------------
        notify("Grand Final Arbiter synthesizing championship solution...", 0.92)
        final_gen = num_generations
        final_leaderboard = generation_leaderboards[final_gen]

        top_3_agent_ids = [e.agent_id for e in final_leaderboard[:3]]
        top_solutions = [current_solutions[a_id] for a_id in top_3_agent_ids if a_id in current_solutions]

        critique_summary = []
        for e in final_leaderboard[:3]:
            critique_summary.extend(e.weaknesses)
            critique_summary.extend(e.strengths)

        arbiter_result = self.arbiter.synthesize(
            task=task,
            top_solutions=top_solutions,
            evaluations_summary=critique_summary,
            total_generations=num_generations,
            total_agents=num_agents
        )

        # -------------------------------------------------------------
        # Compile Metrics & Experiment Analytics (Phase 13 & 17)
        # -------------------------------------------------------------
        notify("Compiling tournament analytics...", 0.97)
        overall_progress = ArenaMetrics.compute_generation_progress(generation_leaderboards)
        overall_tool_experiment = ArenaMetrics.compute_tool_allocation_experiment(final_leaderboard)

        total_runtime = round(time.time() - start_tournament_time, 2)

        tournament_result = {
            "run_id": run_id,
            "timestamp": time.time(),
            "task": task,
            "total_runtime_sec": total_runtime,
            "num_agents": num_agents,
            "num_generations": num_generations,
            "task_analysis": task_analysis,
            "generation_history": generation_history,
            "final_leaderboard": [e.to_dict() for e in final_leaderboard],
            "final_arbiter": arbiter_result,
            "tool_experiment": overall_tool_experiment,
            "progress_analytics": overall_progress
        }

        # Persist run to memory store
        saved_path = self.memory_store.save_run(run_id, tournament_result)
        tournament_result["saved_path"] = saved_path

        notify("Tournament complete! [CHAMPION]", 1.0, tournament_result)
        return tournament_result
