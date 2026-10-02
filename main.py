import argparse
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import validate_config, DEFAULT_MODEL
from arena.controller import ArenaController
from evaluation.benchmark import ArenaBenchmarkRunner


def print_banner():
    banner = r"""
========================================================================
   ___   _____  ______ _   _ _____     ___  ______ _____ _   _   ___   
  / _ \ |  __ \ |  ___| \ | |_   _|   / _ \ | ___ \  ___| \ | | / _ \  
 / /_\ \| |  \/ | |__ |  \| | | |    / /_\ \| |_/ / |__ |  \| |/ /_\ \ 
 |  _  || | __  |  __|| . ` | | |    |  _  ||    /|  __|| . ` ||  _  | 
 | | | || |_\ \ | |___| |\  | | |    | | | || |\ \| |___| |\  || | | | 
 \_| |_/ \____/ \____/\_| \_/ \_/    \_| |_/\_| \_\____/\_| \_/\_| |_/ 
========================================================================
Competitive & Evolutionary Multi-Agent Problem-Solving System
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="AGENT ARENA: Competitive & Evolutionary Multi-Agent System")
    parser.add_argument(
        "--task",
        type=str,
        default="Compare the latest open-source LLMs and recommend one for a startup.",
        help="Task or problem for the Arena to solve"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of agents in the population (e.g. 3 for quick test, up to 20 or 30)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=2,
        help="Number of evolutionary generations (1 to 4)"
    )
    parser.add_argument(
        "--judges",
        type=int,
        default=2,
        help="Number of judges in the multi-judge evaluation panel"
    )
    parser.add_argument(
        "--no-peer",
        action="store_true",
        help="Disable peer critique sampling"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 5-Baseline Comparative Benchmark on the task"
    )
    args = parser.parse_args()

    print_banner()
    cfg_status = validate_config()
    print(f"[*] Configuration Check:")
    print(f"    - Groq Configured:   {'[OK]' if cfg_status['groq_configured'] else '[FAIL]'}")
    print(f"    - Tavily Configured: {'[OK]' if cfg_status['tavily_configured'] else '[FAIL]'}")
    print(f"    - LLM Engine:        {DEFAULT_MODEL}")

    if not cfg_status["groq_configured"]:
        print("[!] Error: GROQ_API_KEY is required in .env or environment.")
        sys.exit(1)

    if args.benchmark:
        print(f"\n[*] Running 5-Baseline Benchmark on task:\n    \"{args.task}\"\n")
        bench = ArenaBenchmarkRunner()
        results = bench.run_benchmark(args.task)
        print("\n" + "=" * 70)
        print(f"{'BASELINE':<40} | {'SCORE':<6} | {'TIME(s)':<8} | {'IMPROV %':<8}")
        print("-" * 70)
        for name, data in results.items():
            print(f"{name:<40} | {data['score']:<6.1f} | {data['latency_sec']:<8.1f} | {data.get('percent_improvement', 0.0):+6.1f}%")
        print("=" * 70)
        return

    print(f"\n[*] Task: \"{args.task}\"")
    print(f"[*] Population: {args.agents} agents | Generations: {args.generations}")
    print(f"[*] Judges: {args.judges} | Peer Evaluation: {'OFF' if args.no_peer else 'ON'}\n")

    def progress_callback(msg: str, pct: float, data=None):
        bar_len = 25
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"[{bar}] {int(pct*100):3d}% | {msg}", flush=True)

    controller = ArenaController(
        num_judges=args.judges,
        peer_sample_k=2 if not args.no_peer else 0
    )

    results = controller.run_tournament(
        task=args.task,
        num_agents=args.agents,
        num_generations=args.generations,
        enable_peer_evaluation=not args.no_peer,
        enable_dynamic_tool_assignment=True,
        progress_callback=progress_callback
    )

    print("\n" + "=" * 75)
    print("🏆 FINAL TOURNAMENT LEADERBOARD")
    print("=" * 75)
    print(f"{'RANK':<5} | {'AGENT':<12} | {'ROLE':<25} | {'TOOLS':<10} | {'SCORE':<6}")
    print("-" * 75)
    for entry in results["final_leaderboard"]:
        tools_str = ",".join(entry["tools"]) if entry["tools"] else "None"
        print(f"{entry['rank']:<5} | {entry['agent_name']:<12} | {entry['role']:<25} | {tools_str:<10} | {entry['score']:<6.1f}")
    print("=" * 75)

    te = results["tool_experiment"]
    print("\n🔬 TOOL ALLOCATION EXPERIMENT")
    print(f"   - Tavily Agents Avg Score:  {te['avg_score_tavily']:.1f} (Count: {te['tavily_agent_count']})")
    print(f"   - No-Tool Agents Avg Score: {te['avg_score_notool']:.1f} (Count: {te['notool_agent_count']})")
    print(f"   - Web Research Lift:        {te['improvement_pct']:+.1f}%")
    print(f"   - Total Web Searches:       {te['total_tavily_calls']}")

    print("\n" + "=" * 75)
    print("🏆 GRAND FINAL ARBITER SYNTHESIZED SOLUTION")
    print("=" * 75)
    print(results["final_arbiter"]["master_solution"])
    print("=" * 75)
    print(f"\n[OK] Run saved to: {results['saved_path']}\n")


if __name__ == "__main__":
    main()
