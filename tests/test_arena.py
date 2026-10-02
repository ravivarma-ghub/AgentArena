import pytest
from agents.base import AgentConfig, AgentSolution, EvaluationResult
from arena.ranking import Ranker
from arena.evolution import EvolutionEngine
from arena.population import create_population
from arena.tool_allocator import ToolAllocator
from evaluation.metrics import ArenaMetrics
from tools.tavily import TavilyTool
from config import validate_config


def test_config_validation():
    cfg = validate_config()
    assert "groq_configured" in cfg
    assert "tavily_configured" in cfg


def test_tavily_tool_initialization():
    tool = TavilyTool()
    assert tool is not None
    # Test fallback or search structure
    res = tool.search("test query", max_results=1)
    assert "results" in res
    assert "formatted_context" in res


def test_population_generation():
    pop3 = create_population(3)
    assert len(pop3) == 3
    # Check diverse roles and strategies
    assert any("tavily" in a.tools for a in pop3)

    pop20 = create_population(20)
    assert len(pop20) == 20


def test_ranking_and_leaderboard():
    sols = [
        AgentSolution(
            agent_id=1,
            agent_name="Agent 01",
            role="Analyst",
            strategy="Evidence-first",
            generation=1,
            task="Test Task",
            answer="Solution 1",
            tools_used=["tavily"],
            latency_sec=1.5,
            total_tokens=200
        ),
        AgentSolution(
            agent_id=2,
            agent_name="Agent 02",
            role="Devil's Advocate",
            strategy="Stress-test",
            generation=1,
            task="Test Task",
            answer="Solution 2",
            tools_used=[],
            latency_sec=1.2,
            total_tokens=180
        )
    ]

    evals = [
        EvaluationResult(
            evaluator_id="Judge 1",
            evaluator_type="judge",
            target_agent_id=1,
            generation=1,
            scores={"correctness": 9.0, "evidence": 9.5},
            overall=9.2,
            strengths=["Good data"],
            weaknesses=["Needs more examples"],
            improvements=["Add tables"]
        ),
        EvaluationResult(
            evaluator_id="Judge 1",
            evaluator_type="judge",
            target_agent_id=2,
            generation=1,
            scores={"correctness": 7.5, "evidence": 6.0},
            overall=7.0,
            strengths=["Good critical thinking"],
            weaknesses=["Lacks data"],
            improvements=["Cite sources"]
        )
    ]

    lb = Ranker.calculate_leaderboard(sols, evals)
    assert len(lb) == 2
    assert lb[0].agent_id == 1
    assert lb[0].rank == 1
    assert lb[0].score == 9.2
    assert lb[1].rank == 2
    assert lb[1].score == 7.0


def test_evolution_engine():
    pop = create_population(4)
    # Mock leaderboard
    sols = [
        AgentSolution(agent_id=a.agent_id, agent_name=a.name, role=a.role, strategy=a.strategy,
                      generation=1, task="Task", answer="...", tools_used=a.tools)
        for a in pop
    ]
    evals = [
        EvaluationResult(evaluator_id="J", evaluator_type="judge", target_agent_id=pop[0].agent_id, generation=1, overall=9.5),
        EvaluationResult(evaluator_id="J", evaluator_type="judge", target_agent_id=pop[1].agent_id, generation=1, overall=8.5),
        EvaluationResult(evaluator_id="J", evaluator_type="judge", target_agent_id=pop[2].agent_id, generation=1, overall=7.0),
        EvaluationResult(evaluator_id="J", evaluator_type="judge", target_agent_id=pop[3].agent_id, generation=1, overall=6.0),
    ]
    lb = Ranker.calculate_leaderboard(sols, evals)
    engine = EvolutionEngine(survival_rate=0.75)
    evolved = engine.evolve_population(pop, lb)
    
    assert len(evolved["population"]) == 4
    assert pop[3].agent_id in evolved["mutated_ids"]


def test_tool_allocation_metrics():
    sols = [
        AgentSolution(agent_id=1, agent_name="A1", role="R", strategy="S", generation=1,
                      task="T", answer="Ans1", tools_used=["tavily"], tavily_calls=2, latency_sec=2.0, total_tokens=500),
        AgentSolution(agent_id=2, agent_name="A2", role="R", strategy="S", generation=1,
                      task="T", answer="Ans2", tools_used=[], tavily_calls=0, latency_sec=1.0, total_tokens=300)
    ]
    evals = [
        EvaluationResult(evaluator_id="J", evaluator_type="judge", target_agent_id=1, generation=1, overall=9.0),
        EvaluationResult(evaluator_id="J", evaluator_type="judge", target_agent_id=2, generation=1, overall=7.5),
    ]
    lb = Ranker.calculate_leaderboard(sols, evals)
    metrics = ArenaMetrics.compute_tool_allocation_experiment(lb)
    
    assert metrics["tavily_agent_count"] == 1
    assert metrics["notool_agent_count"] == 1
    assert metrics["avg_score_tavily"] == 9.0
    assert metrics["avg_score_notool"] == 7.5
    assert metrics["improvement_pct"] == 20.0
    assert metrics["hypothesis_validated"] is True
