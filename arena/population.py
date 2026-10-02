from typing import List, Union
from agents.base import AgentConfig
from agents.solver import SolverAgent
from agents.researcher import ResearcherAgent

POPULATION_TEMPLATES = [
    # General Reasoning
    {"role": "General Problem Solver", "tools": [], "strategy": "First-principles"},
    {"role": "General Problem Solver", "tools": [], "strategy": "Balanced Tradeoff"},
    {"role": "General Problem Solver", "tools": [], "strategy": "Pragmatic Implementation"},
    {"role": "General Problem Solver", "tools": [], "strategy": "Startup-lean"},
    {"role": "General Problem Solver", "tools": [], "strategy": "Evidence-first"},

    # Research (with Tavily)
    {"role": "Research Analyst", "tools": ["tavily"], "strategy": "Evidence-first"},
    {"role": "Research Analyst", "tools": ["tavily"], "strategy": "Balanced Tradeoff"},
    {"role": "Research Analyst", "tools": ["tavily"], "strategy": "Pragmatic Implementation"},
    {"role": "Research Analyst", "tools": ["tavily"], "strategy": "First-principles"},
    {"role": "Research Analyst", "tools": ["tavily"], "strategy": "Startup-lean"},

    # Critical / Devil's Advocate
    {"role": "Devil's Advocate", "tools": [], "strategy": "Stress-test & Risk-analysis"},
    {"role": "Devil's Advocate", "tools": [], "strategy": "Startup-lean"},
    {"role": "Devil's Advocate", "tools": ["tavily"], "strategy": "Stress-test & Risk-analysis"},

    # Technical / Architecture
    {"role": "Technical Architect", "tools": [], "strategy": "Pragmatic Implementation"},
    {"role": "Technical Architect", "tools": ["tavily"], "strategy": "Evidence-first"},
    {"role": "Technical Architect", "tools": [], "strategy": "First-principles"},

    # Creative / Strategic
    {"role": "Creative Innovator", "tools": [], "strategy": "Startup-lean"},
    {"role": "Creative Innovator", "tools": ["tavily"], "strategy": "Evidence-first"},

    # Verification / Auditing
    {"role": "Verification Auditor", "tools": [], "strategy": "First-principles"},
    {"role": "Verification Auditor", "tools": ["tavily"], "strategy": "Evidence-first"},

    # Extended up to 30
    {"role": "Cost & Operations Optimizer", "tools": [], "strategy": "Startup-lean"},
    {"role": "Cost & Operations Optimizer", "tools": ["tavily"], "strategy": "Pragmatic Implementation"},
    {"role": "Technical Architect", "tools": ["tavily"], "strategy": "Balanced Tradeoff"},
    {"role": "General Problem Solver", "tools": [], "strategy": "Stress-test & Risk-analysis"},
    {"role": "Research Analyst", "tools": ["tavily"], "strategy": "Evidence-first"},
    {"role": "Devil's Advocate", "tools": [], "strategy": "Stress-test & Risk-analysis"},
    {"role": "Creative Innovator", "tools": [], "strategy": "Pragmatic Implementation"},
    {"role": "Verification Auditor", "tools": ["tavily"], "strategy": "First-principles"},
    {"role": "Cost & Operations Optimizer", "tools": [], "strategy": "Balanced Tradeoff"},
    {"role": "General Problem Solver", "tools": ["tavily"], "strategy": "Evidence-first"},
]


def create_population(
    population_size: int = 3,
    groq_api_key: str = "",
    tavily_api_key: str = ""
) -> List[Union[SolverAgent, ResearcherAgent]]:
    """
    Generate an agent population of specified size with balanced diversity.
    Supports sizes from 3 (fast test) to 30 (massive arena).
    """
    agents: List[Union[SolverAgent, ResearcherAgent]] = []
    
    # Select configs
    if population_size == 3:
        # Matches exact specification: A1 (NoTool), A2 (Tavily), A3 (NoTool)
        selected_templates = [
            POPULATION_TEMPLATES[0],   # General Problem Solver (No tools)
            POPULATION_TEMPLATES[5],   # Research Analyst (Tavily)
            POPULATION_TEMPLATES[10],  # Devil's Advocate (No tools)
        ]
    elif population_size <= len(POPULATION_TEMPLATES):
        # Interleave tool and non-tool templates to ensure balanced population
        tool_tmpls = [t for t in POPULATION_TEMPLATES if "tavily" in t["tools"]]
        notool_tmpls = [t for t in POPULATION_TEMPLATES if "tavily" not in t["tools"]]
        interleaved = []
        for i in range(max(len(tool_tmpls), len(notool_tmpls))):
            if i < len(notool_tmpls):
                interleaved.append(notool_tmpls[i])
            if i < len(tool_tmpls):
                interleaved.append(tool_tmpls[i])
        selected_templates = interleaved[:population_size]
    else:
        selected_templates = POPULATION_TEMPLATES * (population_size // len(POPULATION_TEMPLATES) + 1)
        selected_templates = selected_templates[:population_size]

    for idx, tmpl in enumerate(selected_templates, 1):
        config = AgentConfig(
            id=idx,
            name=f"Agent {idx:02d}",
            role=tmpl["role"],
            tools=list(tmpl["tools"]),
            strategy=tmpl["strategy"]
        )

        if "tavily" in config.tools:
            agent = ResearcherAgent(
                config=config,
                groq_api_key=groq_api_key,
                tavily_api_key=tavily_api_key
            )
        else:
            agent = SolverAgent(
                config=config,
                groq_api_key=groq_api_key
            )
        agents.append(agent)

    return agents
