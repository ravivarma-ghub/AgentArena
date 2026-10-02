PEER_CRITIC_PROMPT = """You are {critic_name}, a peer reviewer evaluating another agent's solution.
Your role: {critic_role}
Your perspective: {critic_strategy}

TASK:
\"\"\"{task}\"\"\"

PEER SOLUTION TO REVIEW:
\"\"\"{solution}\"\"\"

Critique this solution thoroughly from your specialized perspective.
Identify:
1. What does this solution do well?
2. What critical gaps, blind spots, or technical inaccuracies exist?
3. What concrete modifications would make this solution world-class?

Respond with a JSON object:
{{
  "peer_score": <float 1.0-10.0>,
  "critique_summary": "1-2 sentence overall peer appraisal",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "actionable_tips": ["...", "..."]
}}
"""
