# 🚀 AGENT ARENA: Competitive & Evolutionary Multi-Agent Problem-Solving System

Agent Arena is an autonomous, competitive, and evolutionary problem-solving system powered by **Groq** high-speed LLM inference and **Tavily** web search. 

Instead of asking a single LLM to provide a static answer, Agent Arena generates a diverse population of specialized agents (Reasoning, Empirical Research, Devil's Advocate, Technical Architecture, Creative Strategy, Verification). Agents independently solve the problem, critique each other, receive multi-criteria feedback from a multi-judge panel, evolve over successive generations, and finally submit their solutions to a **Grand Final Arbiter** that synthesizes a championship solution superior to any single agent's response.

---

## 🏗️ Architecture & Evolutionary Lifecycle

```
                         USER TASK
                            │
                            ▼
                     ┌─────────────┐
                     │ Arena       │
                     │ Controller  │
                     └──────┬──────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
       Agent Pool       Agent Pool       Agent Pool
       No Tools         Tavily           Tavily
           │                │                │
           └────────────────┼────────────────┘
                            ▼
                    PEER EVALUATION
                            │
                            ▼
                    MULTI-JUDGE PANEL
                            │
                            ▼
                       RANKING
                            │
                            ▼
                     FEEDBACK LOOP
                            │
                            ▼
                     GENERATION 2 (Evolution & Mutation)
                            │
                            ▼
                     GENERATION 3
                            │
                            ▼
                     GENERATION 4
                            │
                            ▼
                    FINAL ARBITER
                            │
                            ▼
                    🏆 BEST ANSWER
```

---

## 🌟 Key Features Across 17 Phases

1. **Diverse Agent Population (Phase 1-4)**: Scalable from 3 to 30 agents with distinct archetypes (General Problem Solver, Research Analyst, Devil's Advocate, Technical Architect, Creative Innovator, Verification Auditor, Cost & Operations Optimizer).
2. **Selective & Intelligent Web Research (Phase 2 & 14)**: Research agents dynamically determine *"Do I actually need web research?"* before issuing targeted queries to Tavily.
3. **Independent Solving (Phase 5)**: Solvers produce unpolluted, independent perspectives without seeing peer outputs in initial rounds.
4. **Multi-Judge Panel (Phase 6 & 7)**: Independent evaluation panel with varied personas (Strict Academic, Startup CTO, Balanced Tech Lead) grading 6 criteria: *Correctness, Completeness, Reasoning, Evidence, Clarity, Practicality*.
5. **Sampled Peer Evaluation (Phase 8)**: Prevents $O(N^2)$ explosion via random peer critique sampling.
6. **Dynamic Ranking & Leaderboard (Phase 9)**: Real-time score aggregation across judges and peer reviews.
7. **Iterative Feedback & Improvement (Phase 10 & 11)**: Agents incorporate strengths, weaknesses, and concrete recommendations into Generation $N+1$.
8. **Survival & Mutation (Phase 12)**: Top performers survive; bottom-tier agents mutate strategies and tool configurations.
9. **Empirical Tool Allocation Experiment (Phase 13)**: Mathematically quantifies the performance lift of Tavily-equipped agents vs. static LLM reasoning:
   $$\text{Quality Lift} = \frac{\text{Score}_{\text{Tavily}} - \text{Score}_{\text{NoTool}}}{\text{Score}_{\text{NoTool}}} \times 100\%$$
10. **Supreme Final Arbiter (Phase 15)**: Synthesizes top candidates, resolving remaining blind spots into a production-grade master solution.
11. **Interactive Streamlit Dashboard (Phase 16)**: Full interactive web application with leaderboards, generation history, tool comparisons, and agent inspectors.
12. **5-Baseline Benchmark Suite (Phase 17)**: Side-by-side comparison across Single Agent, Voting Ensemble, Arena (No Tools), Arena (+ Tavily), and Evolutionary Arena.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/Agent_Arena.git
cd Agent_Arena
pip install -r requirements.txt
```

### 2. Environment Configuration

Create or update `.env` in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
DEFAULT_MODEL=qwen/qwen3.8-27b
```

---

## 🖥️ Running the Streamlit Frontend

To launch the web dashboard:

```bash
streamlit run app.py
```
*(Or alternatively: `streamlit run ui/dashboard.py`)*

The UI enables you to:
- Enter custom tasks or select presets (e.g. LLM comparison, RAG system design, binary search).
- Select population size (3 to 30 agents) and evolutionary generations (1 to 4).
- View live tournament progress bars.
- Inspect the **Grand Arbiter Master Solution**.
- Track evolutionary score charts across generations.
- Analyze the **Tavily Tool Lift Experiment** and review individual agent cards.

---

## 💻 Running via CLI

### Run a Standard Tournament
```bash
python main.py --task "Compare the latest open-source LLMs and recommend one for a startup." --agents 3 --generations 2
```

### Run the 5-Baseline Benchmark
```bash
python main.py --benchmark --task "Compare the latest open-source LLMs and recommend one for a startup."
```

---

## 🌐 Running the FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation available at: `http://localhost:8000/docs`.

---

## 🧪 Testing

Run automated tests:

```bash
python -m pytest tests/ -v
```

---

## 📁 Repository Structure

```
Agent_Arena/
├── agents/
│   ├── base.py          # Core AgentConfig, AgentSolution, BaseAgent
│   ├── solver.py        # Independent solver & feedback improver
│   ├── researcher.py    # Tavily tool integration & search decision logic
│   ├── critic.py        # Peer review & critique agent
│   ├── judge.py         # Multi-criteria scoring judge
│   └── arbiter.py       # Master synthesis arbiter
├── arena/
│   ├── controller.py    # Master tournament orchestrator
│   ├── population.py    # Population templates & generator
│   ├── tournament.py    # Peer review sampling
│   ├── ranking.py       # Leaderboard & score aggregation
│   ├── evolution.py     # Survival selection & mutation engine
│   └── tool_allocator.py# Task analyzer & dynamic tool allocator
├── tools/
│   ├── tavily.py        # Tavily Search API wrapper
│   └── tavily_tool.py   # Compatibility alias
├── evaluation/
│   ├── evaluator.py     # Multi-judge panel coordinator
│   ├── metrics.py       # Tool experiment & evolutionary metrics
│   └── benchmark.py     # 5-baseline comparative benchmark
├── memory/
│   └── store.py         # Persistence for tournament runs
├── prompts/
│   ├── solver.py        # Solver & role prompts
│   ├── judge.py         # Multi-criteria evaluation rubric
│   ├── critic.py        # Peer evaluation prompts
│   ├── arbiter.py       # Final arbiter synthesis prompts
│   └── prompts.py       # Barrel export
├── ui/
│   └── dashboard.py     # Streamlit frontend application
├── api/
│   └── main.py          # FastAPI REST endpoints
├── tests/
│   └── test_arena.py    # Pytest test suite
├── app.py               # Streamlit application launcher
├── main.py              # CLI entry point
├── config.py            # Global configuration & environment loader
├── requirements.txt     # Python dependencies
└── .env                 # API keys
```
