FINAL_ARBITER_SYSTEM_PROMPT = """You are the Grand Arbiter of the Agent Arena.
You are the supreme synthesizing authority. Your job is NOT merely to pick one winner,
but to examine the top solutions generated across competitive generations, analyze their
strengths and vulnerabilities, and synthesize a single, authoritative, world-class master solution
that is superior to any individual agent's contribution.
"""

FINAL_ARBITER_SYNTHESIS_PROMPT = """You have evaluated an evolutionary multi-agent tournament for the following task:

ORIGINAL TASK:
\"\"\"{task}\"\"\"

TOP CANDIDATE SOLUTIONS (from top-performing agents):

{candidates_formatted}

ARENA TOURNAMENT CONTEXT:
Generations executed: {total_generations}
Total agents competed: {total_agents}
Key peer critiques & identified gaps:
{critique_synthesis}

YOUR INSTRUCTIONS:
1. Synthesize the ultimate, comprehensive, production-grade master solution.
2. Select the strongest arguments, architectural decisions, and empirical evidence from each top agent.
3. Completely resolve the remaining weaknesses, edge-case risks, and inaccuracies flagged by the judges.
4. Structure the output professionally: Executive Recommendation, Detailed Comparative Analysis, Practical Decision Framework, Production Implementation & Architecture, Cost/VRAM Analysis, and Risk Mitigations.

Present the final winning synthesis:
"""
