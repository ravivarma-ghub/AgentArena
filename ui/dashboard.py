import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import (
    validate_config,
    reload_keys,
    DEFAULT_MODEL,
    GROQ_API_KEY,
    TAVILY_API_KEY,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_NUM_GENERATIONS
)
from arena.controller import ArenaController
from evaluation.benchmark import ArenaBenchmarkRunner
from memory.store import ArenaMemoryStore

# Page Config
st.set_page_config(
    page_title="AGENT ARENA | Evolutionary Multi-Agent System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F00, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #888888;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding-top: 8px;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Tournament Settings")

# API Keys Management in Sidebar
cfg_status = validate_config()
with st.sidebar.expander("🔑 API Key Credentials", expanded=not (cfg_status["groq_configured"] and cfg_status["tavily_configured"])):
    user_groq_key = st.text_input("Groq API Key", value=GROQ_API_KEY, type="password",
                                  help="Automatically read from .env. You can also paste or override here.")
    user_tavily_key = st.text_input("Tavily API Key", value=TAVILY_API_KEY, type="password",
                                    help="Automatically read from .env. You can also paste or override here.")
    if user_groq_key != GROQ_API_KEY or user_tavily_key != TAVILY_API_KEY:
        reload_keys(user_groq_key, user_tavily_key)
        cfg_status = validate_config(user_groq_key, user_tavily_key)

# Header
col_header, col_status = st.columns([3, 1])
with col_header:
    st.markdown('<p class="main-title">⚡ AGENT ARENA</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Competitive & Evolutionary Multi-Agent Problem-Solving System</p>', unsafe_allow_html=True)

with col_status:
    groq_badge = "🟢 Groq Ready" if cfg_status["groq_configured"] else "🔴 Groq Missing"
    tavily_badge = "🟢 Tavily Ready" if cfg_status["tavily_configured"] else "🟡 Tavily Missing"
    st.markdown(f"**Status:** {groq_badge} | {tavily_badge}")
    st.caption(f"Engine: `{DEFAULT_MODEL}`")

preset_tasks = {
    "Open-Source LLMs (Research)": "Compare the latest open-source LLMs and recommend one for a startup.",
    "System Design (Architectural)": "Design a high-throughput, production-grade RAG system with multi-tenancy, vector caching, and fallback reranking.",
    "Algorithm (Static Reasoning)": "Explain binary search, analyze time/space complexity, and write a bug-free Python implementation handling duplicate values.",
    "Custom Task": ""
}

selected_preset = st.sidebar.selectbox("Preset Tasks", list(preset_tasks.keys()))
if selected_preset != "Custom Task":
    task_input = st.sidebar.text_area("Task Description", value=preset_tasks[selected_preset], height=110)
else:
    task_input = st.sidebar.text_area("Task Description", value="Compare the latest open-source LLMs and recommend one for a startup.", height=110)

num_agents = st.sidebar.slider("Population Size (Agents)", min_value=3, max_value=30, value=3, step=1,
                              help="Recommended: 3 for instant tests, 10-20 for full competitive diversity.")
num_generations = st.sidebar.slider("Evolutionary Generations", min_value=1, max_value=4, value=2, step=1,
                                   help="Generations of solve -> evaluate -> feedback -> improve.")
num_judges = st.sidebar.slider("Judge Panel Size", min_value=1, max_value=4, value=2, step=1,
                              help="Number of multi-persona judges to reduce bias.")
enable_peer = st.sidebar.checkbox("Enable Peer Critiques", value=True,
                                  help="Agents critique peer solutions with random sampling.")
enable_dynamic_tools = st.sidebar.checkbox("Dynamic Tool Allocator", value=True,
                                          help="Task Analyzer determines live research requirements.")

# Run History loader
memory = ArenaMemoryStore()
saved_runs = memory.list_runs()
if saved_runs:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Past Runs")
    run_options = {f"{r['run_id']} ({r['generations']}G)": r['run_id'] for r in saved_runs[:8]}
    selected_run_label = st.sidebar.selectbox("Load Previous Run", ["-- Current Session --"] + list(run_options.keys()))
    if selected_run_label != "-- Current Session --":
        loaded_data = memory.load_run(run_options[selected_run_label])
        if loaded_data:
            st.session_state["tournament_results"] = loaded_data

launch_btn = st.sidebar.button("🚀 Launch Arena Tournament", type="primary", use_container_width=True)
benchmark_btn = st.sidebar.button("🧪 Run 5-Baseline Benchmark", use_container_width=True)

# Main Tournament Execution
if launch_btn and task_input.strip():
    if not cfg_status["groq_configured"]:
        st.error("❌ Groq API key is missing! Please configure it in .env or the sidebar credentials.")
    else:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg: str, pct: float, data=None):
            progress_bar.progress(min(1.0, max(0.0, pct)))
            status_text.info(f"⏳ **{int(pct * 100)}%** — {msg}")

        controller = ArenaController(
            groq_api_key=user_groq_key or GROQ_API_KEY,
            tavily_api_key=user_tavily_key or TAVILY_API_KEY,
            num_judges=num_judges,
            peer_sample_k=2 if enable_peer else 0
        )

        with st.spinner("Tournament in progress..."):
            results = controller.run_tournament(
                task=task_input,
                num_agents=num_agents,
                num_generations=num_generations,
                enable_peer_evaluation=enable_peer,
                enable_dynamic_tool_assignment=enable_dynamic_tools,
                progress_callback=update_progress
            )

        progress_bar.progress(1.0)
        status_text.success("🎉 **Tournament Completed Successfully!**")
        st.session_state["tournament_results"] = results
        st.session_state.pop("benchmark_results", None)

elif benchmark_btn and task_input.strip():
    if not cfg_status["groq_configured"]:
        st.error("❌ Groq API key is missing! Please configure it in .env or the sidebar credentials.")
    else:
        with st.spinner("Running 5-Baseline Benchmark across architectures..."):
            bench = ArenaBenchmarkRunner(
                groq_api_key=user_groq_key or GROQ_API_KEY,
                tavily_api_key=user_tavily_key or TAVILY_API_KEY
            )
            bench_results = bench.run_benchmark(task_input)
            st.session_state["benchmark_results"] = bench_results
            st.session_state.pop("tournament_results", None)
        st.success("✅ Benchmark Completed!")


# Display Results
if "tournament_results" in st.session_state:
    res = st.session_state["tournament_results"]
    task_desc = res.get("task", "")
    runtime = res.get("total_runtime_sec", 0)
    final_lb = res.get("final_leaderboard", [])
    arbiter_data = res.get("final_arbiter", {})
    tool_exp = res.get("tool_experiment", {})
    gen_hist = res.get("generation_history", {})
    task_analysis = res.get("task_analysis", {})

    st.markdown("---")
    
    # Quick Meta Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Agents", res.get("num_agents", len(final_lb)))
    with m2:
        st.metric("Generations", res.get("num_generations", len(gen_hist)))
    with m3:
        st.metric("Tournament Runtime", f"{runtime}s")
    with m4:
        winner_name = final_lb[0]["agent_name"] if final_lb else "N/A"
        winner_score = final_lb[0]["score"] if final_lb else 0.0
        st.metric("Top Agent", f"{winner_name} ({winner_score:.1f})")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Grand Arbiter Solution",
        "📊 Leaderboard & Evolution",
        "🔬 Tool Experiment (Tavily Lift)",
        "👥 Agent Inspector",
        "🎯 Task & Strategy Analysis"
    ])

    with tab1:
        st.subheader("🏆 Supreme Grand Arbiter: Master Synthesized Solution")
        st.caption("The Final Arbiter synthesizes the top candidate solutions, eliminates identified flaws, and produces the championship answer.")
        
        top_cands = arbiter_data.get("top_candidates", [])
        if top_cands:
            cand_cols = st.columns(len(top_cands))
            for i, cand in enumerate(top_cands):
                with cand_cols[i]:
                    tools_s = ", ".join(cand.get("tools_used", [])) or "No Tools"
                    st.info(f"**Rank {i+1}: {cand.get('agent_name')}** ({cand.get('role')})\n\nScore: **{cand.get('score', 0):.1f}** | Tools: `{tools_s}`")

        st.markdown("### Master Championship Answer")
        st.markdown(arbiter_data.get("master_solution", "No solution recorded."))

    with tab2:
        st.subheader("📊 Tournament Leaderboard & Evolutionary Progression")
        
        # Generation Selector
        gen_keys = sorted([int(k) for k in gen_hist.keys()])
        selected_gen = st.radio("Select Generation View", [f"Generation {g}" for g in gen_keys], horizontal=True)
        gen_idx = int(selected_gen.split()[-1])
        curr_gen_data = gen_hist.get(str(gen_idx), gen_hist.get(gen_idx, {}))
        gen_lb = curr_gen_data.get("leaderboard", final_lb)

        # Leaderboard Table
        lb_rows = []
        for entry in gen_lb:
            lb_rows.append({
                "Rank": entry.get("rank"),
                "Agent": entry.get("agent_name"),
                "Role": entry.get("role"),
                "Strategy": entry.get("strategy"),
                "Tools": ", ".join(entry.get("tools", [])) or "None",
                "Score": entry.get("score"),
                "Latency (s)": entry.get("latency_sec"),
                "Tokens": entry.get("total_tokens"),
                "Tavily Calls": entry.get("tavily_calls", 0)
            })
        df_lb = pd.DataFrame(lb_rows)
        st.dataframe(df_lb, use_container_width=True, hide_index=True)

        # Progression Chart across generations
        if len(gen_keys) > 1:
            st.markdown("#### Score Progression Across Generations")
            prog_data = []
            for g in gen_keys:
                g_data = gen_hist.get(str(g), gen_hist.get(g, {}))
                for item in g_data.get("leaderboard", []):
                    prog_data.append({
                        "Generation": f"Gen {g}",
                        "Agent": item.get("agent_name"),
                        "Score": item.get("score"),
                        "Role": item.get("role")
                    })
            df_prog = pd.DataFrame(prog_data)
            fig_prog = px.line(df_prog, x="Generation", y="Score", color="Agent", markers=True,
                               title="Agent Fitness Evolution Across Generations")
            fig_prog.update_layout(yaxis_range=[5, 10])
            st.plotly_chart(fig_prog, use_container_width=True)

    with tab3:
        st.subheader("🔬 Tool Allocation Experiment: Scientific Analysis")
        st.write("Does web research with Tavily genuinely improve problem-solving quality over foundational LLM reasoning?")

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("Tavily Agents Avg Score", f"{tool_exp.get('avg_score_tavily', 0):.2f}",
                      help=f"Sample size: {tool_exp.get('tavily_agent_count', 0)} agents")
        with col_t2:
            st.metric("No-Tool Agents Avg Score", f"{tool_exp.get('avg_score_notool', 0):.2f}",
                      help=f"Sample size: {tool_exp.get('notool_agent_count', 0)} agents")
        with col_t3:
            lift = tool_exp.get('improvement_pct', 0.0)
            st.metric("Empirical Quality Lift", f"{lift:+.1f}%",
                      delta=f"{lift:+.1f}% vs baseline",
                      delta_color="normal")

        # Bar chart comparison
        fig_tool = go.Figure(data=[
            go.Bar(name='Average Score', x=['Tavily Research Agents', 'No-Tool Static Agents'],
                   y=[tool_exp.get('avg_score_tavily', 0), tool_exp.get('avg_score_notool', 0)],
                   marker_color=['#00CC96', '#636EFA']),
            go.Bar(name='Avg Latency (s)', x=['Tavily Research Agents', 'No-Tool Static Agents'],
                   y=[tool_exp.get('avg_latency_tavily_sec', 0), tool_exp.get('avg_latency_notool_sec', 0)],
                   marker_color=['#FFA15A', '#AB63FA'])
        ])
        fig_tool.update_layout(barmode='group', title="Tavily vs No-Tool: Score & Latency Comparison")
        st.plotly_chart(fig_tool, use_container_width=True)

        st.markdown(f"""
        - **Total Tavily Live Search Invocations:** `{tool_exp.get('total_tavily_calls', 0)}`
        - **Total Tokens Consumed (Tavily Agents):** `{tool_exp.get('total_tokens_tavily', 0):,}`
        - **Total Tokens Consumed (No-Tool Agents):** `{tool_exp.get('total_tokens_notool', 0):,}`
        """)

    with tab4:
        st.subheader("👥 Detailed Agent Inspector")
        agent_names = [e["agent_name"] for e in final_lb]
        selected_agent_name = st.selectbox("Inspect Agent Details", agent_names)
        
        agent_entry = next((e for e in final_lb if e["agent_name"] == selected_agent_name), None)
        if agent_entry:
            col_a1, col_a2 = st.columns([1, 2])
            with col_a1:
                st.markdown(f"### {agent_entry['agent_name']}")
                st.markdown(f"**Role:** {agent_entry['role']}")
                st.markdown(f"**Strategy:** `{agent_entry['strategy']}`")
                st.markdown(f"**Tools:** `{', '.join(agent_entry['tools']) or 'None'}`")
                st.markdown(f"**Final Score:** `{agent_entry['score']:.1f} / 10.0`")
                
                # Radar or bar of criteria scores
                dim_scores = agent_entry.get("dimension_scores", {})
                if dim_scores:
                    st.markdown("**Criteria Breakdown:**")
                    for k, v in dim_scores.items():
                        st.progress(min(1.0, v / 10.0), text=f"{k.capitalize()}: {v:.1f}")

            with col_a2:
                # Strengths and Weaknesses
                st.markdown("#### Strengths Recognized by Evaluators")
                for s in agent_entry.get("strengths", []):
                    st.success(f"✓ {s}")

                st.markdown("#### Weaknesses & Identified Blindspots")
                for w in agent_entry.get("weaknesses", []):
                    st.warning(f"⚠ {w}")

            # Evolution Across Generations for this Agent
            st.markdown("---")
            st.markdown(f"#### Generated Solutions by {agent_entry['agent_name']}")
            for g in gen_keys:
                g_data = gen_hist.get(str(g), gen_hist.get(g, {}))
                sol_entry = next((s for s in g_data.get("solutions", []) if s.get("agent_id") == agent_entry["agent_id"]), None)
                if sol_entry:
                    with st.expander(f"Generation {g} Solution (Latency: {sol_entry.get('latency_sec', 0):.1f}s | Tokens: {sol_entry.get('total_tokens', 0)})"):
                        st.markdown(sol_entry.get("answer", ""))

    with tab5:
        st.subheader("🎯 Task Analyzer & Dynamic Allocation")
        if task_analysis:
            st.json(task_analysis)
        else:
            st.info("Dynamic Tool Assignment was disabled or handled by standard allocation.")

elif "benchmark_results" in st.session_state:
    st.subheader("🧪 5-Baseline System Benchmark (Phase 17)")
    st.caption("Demonstrating whether multi-agent competition and evolution outperform single agents and simple voting ensembles.")

    b_data = st.session_state["benchmark_results"]
    rows = []
    for b_name, b_vals in b_data.items():
        rows.append({
            "Baseline Architecture": b_name,
            "Quality Score": b_vals.get("score"),
            "Latency (s)": b_vals.get("latency_sec"),
            "Tokens": b_vals.get("total_tokens"),
            "Tavily Searches": b_vals.get("tavily_calls"),
            "Lift vs Single Agent": f"{b_vals.get('percent_improvement', 0.0):+.1f}%"
        })
    df_bench = pd.DataFrame(rows)
    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    fig_bench = px.bar(
        df_bench,
        x="Baseline Architecture",
        y="Quality Score",
        color="Quality Score",
        title="Quality Score Comparison Across 5 Baselines",
        range_y=[5, 10],
        text_auto=".1f"
    )
    st.plotly_chart(fig_bench, use_container_width=True)

else:
    # Landing Guide
    st.info("👈 Select a task in the sidebar and click **'Launch Arena Tournament'** to start the evolutionary competition.")

    st.markdown("""
    ### How AGENT ARENA Operates:
    1. **Dynamic Task Analysis**: Evaluates whether live web research is required.
    2. **Diverse Agent Population**: Generates agents with distinct archetypes (Analyst, Devil's Advocate, Technical Architect, Creative Innovator, FinOps Optimizer).
    3. **Independent Solving**: Round 1 agents solve independently without contamination.
    4. **Multi-Judge Panel & Peer Review**: Multi-persona judges & peer sampling evaluate solutions across 6 criteria.
    5. **Survival & Mutation**: Low-performing strategies evolve toward winning behaviors.
    6. **Iterative Feedback Loop**: Agents receive targeted critiques and produce improved generations.
    7. **Grand Final Arbiter**: Synthesizes the top answers into a single, dominant master solution.
    """)
