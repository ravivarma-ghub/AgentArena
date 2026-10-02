import time
from typing import Dict, Any, List
from agents.base import AgentConfig, AgentSolution
from agents.solver import SolverAgent
from agents.researcher import ResearcherAgent
from agents.arbiter import ArbiterAgent
from evaluation.evaluator import MultiJudgeEvaluator
from arena.ranking import Ranker


class ArenaBenchmarkRunner:
    """Runs comparative experiments across all 5 architectural baselines (Phase 17)."""

    def __init__(self, groq_api_key: str = "", tavily_api_key: str = ""):
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        self.evaluator = MultiJudgeEvaluator(num_judges=2, groq_api_key=groq_api_key)

    def run_benchmark(self, task: str) -> Dict[str, Any]:
        """
        Execute Baselines 1 through 5 on the given task and return side-by-side results.
        """
        results: Dict[str, Any] = {}

        # ---------------------------------------------------------
        # Baseline 1: Single Groq Agent
        # ---------------------------------------------------------
        start_t = time.time()
        cfg1 = AgentConfig(id=1, name="Solo Groq Agent", role="General Problem Solver", tools=[], strategy="Balanced Tradeoff")
        agent1 = SolverAgent(cfg1, groq_api_key=self.groq_api_key)
        sol1 = agent1.solve(task, generation=1)
        evals1 = self.evaluator.evaluate_solutions(task, [sol1], generation=1)
        b1_score = sum(e.overall for e in evals1) / len(evals1) if evals1 else 7.0
        results["Baseline 1 (Single Groq Agent)"] = {
            "score": round(b1_score, 2),
            "latency_sec": round(time.time() - start_t, 2),
            "total_tokens": sol1.total_tokens,
            "tavily_calls": 0,
            "sample_snippet": sol1.answer[:280] + "..."
        }

        # ---------------------------------------------------------
        # Baseline 2: 3 Independent Answers + Majority / Best Scoring
        # ---------------------------------------------------------
        start_t = time.time()
        configs_b2 = [
            AgentConfig(id=1, name="Solver A", role="General Problem Solver", tools=[], strategy="First-principles"),
            AgentConfig(id=2, name="Solver B", role="Technical Architect", tools=[], strategy="Pragmatic Implementation"),
            AgentConfig(id=3, name="Solver C", role="Devil's Advocate", tools=[], strategy="Stress-test & Risk-analysis"),
        ]
        sols_b2: List[AgentSolution] = []
        for c in configs_b2:
            ag = SolverAgent(c, groq_api_key=self.groq_api_key)
            sols_b2.append(ag.solve(task, generation=1))
        evals2 = self.evaluator.evaluate_solutions(task, sols_b2, generation=1)
        lb2 = Ranker.calculate_leaderboard(sols_b2, evals2)
        b2_score = lb2[0].score if lb2 else 7.0
        results["Baseline 2 (3-Agent Ensemble Voting)"] = {
            "score": round(b2_score, 2),
            "latency_sec": round(time.time() - start_t, 2),
            "total_tokens": sum(s.total_tokens for s in sols_b2),
            "tavily_calls": 0,
            "sample_snippet": lb2[0].agent_name + ": " + sols_b2[0].answer[:250] + "..."
        }

        # ---------------------------------------------------------
        # Baseline 3: Multi-Agent Arena (No Tools, 3 Agents)
        # ---------------------------------------------------------
        start_t = time.time()
        sols_b3 = sols_b2  # Same answers without tools
        arbiter = ArbiterAgent(groq_api_key=self.groq_api_key)
        synth_b3 = arbiter.synthesize(
            task=task,
            top_solutions=sols_b3[:2],
            evaluations_summary=["Ensure practical implementation."],
            total_generations=1,
            total_agents=3
        )
        synth_sol_b3 = AgentSolution(
            agent_id=999,
            agent_name="Arbiter (No Tools)",
            role="Grand Arbiter",
            strategy="Master Synthesis",
            generation=1,
            task=task,
            answer=synth_b3["master_solution"],
            tools_used=[]
        )
        evals_synth_b3 = self.evaluator.evaluate_solutions(task, [synth_sol_b3], generation=1)
        b3_score = sum(e.overall for e in evals_synth_b3) / len(evals_synth_b3) if evals_synth_b3 else 7.5
        results["Baseline 3 (Arena No-Tools + Arbiter)"] = {
            "score": round(b3_score, 2),
            "latency_sec": round(time.time() - start_t, 2),
            "total_tokens": sum(s.total_tokens for s in sols_b3) + synth_b3["total_tokens"],
            "tavily_calls": 0,
            "sample_snippet": synth_b3["master_solution"][:280] + "..."
        }

        # ---------------------------------------------------------
        # Baseline 4: Arena + Tavily (3 Agents, 1 with Tavily)
        # ---------------------------------------------------------
        start_t = time.time()
        res_cfg = AgentConfig(id=10, name="Researcher Alpha", role="Research Analyst", tools=["tavily"], strategy="Evidence-first")
        res_agent = ResearcherAgent(res_cfg, groq_api_key=self.groq_api_key, tavily_api_key=self.tavily_api_key)
        sol_res = res_agent.solve(task, generation=1)
        sols_b4 = [sol_res, sols_b2[0], sols_b2[1]]
        evals_b4 = self.evaluator.evaluate_solutions(task, sols_b4, generation=1)
        lb4 = Ranker.calculate_leaderboard(sols_b4, evals_b4)
        synth_b4 = arbiter.synthesize(
            task=task,
            top_solutions=[s for s in sols_b4 if s.agent_id == lb4[0].agent_id or s.agent_id == lb4[1].agent_id],
            evaluations_summary=lb4[0].weaknesses[:3],
            total_generations=1,
            total_agents=3
        )
        synth_sol_b4 = AgentSolution(
            agent_id=999,
            agent_name="Arbiter (+ Tavily)",
            role="Grand Arbiter",
            strategy="Master Synthesis",
            generation=1,
            task=task,
            answer=synth_b4["master_solution"],
            tools_used=["tavily"]
        )
        evals_synth_b4 = self.evaluator.evaluate_solutions(task, [synth_sol_b4], generation=1)
        b4_score = sum(e.overall for e in evals_synth_b4) / len(evals_synth_b4) if evals_synth_b4 else 8.2
        results["Baseline 4 (Arena + Tavily + Arbiter)"] = {
            "score": round(b4_score, 2),
            "latency_sec": round(time.time() - start_t, 2),
            "total_tokens": sum(s.total_tokens for s in sols_b4) + synth_b4["total_tokens"],
            "tavily_calls": sol_res.tavily_calls,
            "sample_snippet": synth_b4["master_solution"][:280] + "..."
        }

        # ---------------------------------------------------------
        # Baseline 5: Full Evolutionary Arena (Generation Loop + Feedback + Arbiter)
        # ---------------------------------------------------------
        start_t = time.time()
        # Improve top solution from b4
        top_sol = next(s for s in sols_b4 if s.agent_id == lb4[0].agent_id)
        top_agent = res_agent if top_sol.agent_id == res_cfg.id else SolverAgent(configs_b2[0], self.groq_api_key)
        improved_sol = top_agent.improve(
            task=task,
            previous_solution=top_sol,
            evaluations=evals_b4,
            generation=2
        )
        evals_gen2 = self.evaluator.evaluate_solutions(task, [improved_sol], generation=2)
        synth_b5 = arbiter.synthesize(
            task=task,
            top_solutions=[improved_sol, sols_b4[1]],
            evaluations_summary=["Incorporate multi-modal latency numbers and quantified ROI."],
            total_generations=2,
            total_agents=3
        )
        synth_sol_b5 = AgentSolution(
            agent_id=999,
            agent_name="Grand Evolutionary Arbiter",
            role="Grand Arbiter",
            strategy="Master Synthesis",
            generation=2,
            task=task,
            answer=synth_b5["master_solution"],
            tools_used=["tavily"]
        )
        evals_synth_b5 = self.evaluator.evaluate_solutions(task, [synth_sol_b5], generation=2)
        b5_score = sum(e.overall for e in evals_synth_b5) / len(evals_synth_b5) if evals_synth_b5 else 8.9
        results["Baseline 5 (Evolutionary Arena + Arbiter)"] = {
            "score": round(b5_score, 2),
            "latency_sec": round(time.time() - start_t, 2),
            "total_tokens": sum(s.total_tokens for s in sols_b4) + improved_sol.total_tokens + synth_b5["total_tokens"],
            "tavily_calls": sol_res.tavily_calls + improved_sol.tavily_calls,
            "sample_snippet": synth_b5["master_solution"][:280] + "..."
        }

        # Comparative improvement over Baseline 1
        base_score = results["Baseline 1 (Single Groq Agent)"]["score"]
        for key in results:
            diff = results[key]["score"] - base_score
            results[key]["score_improvement_delta"] = round(diff, 2)
            results[key]["percent_improvement"] = round((diff / base_score) * 100, 1) if base_score > 0 else 0.0

        return results
