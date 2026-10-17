import time
import html

import pandas as pd
import PyPDF2
import plotly.express as px
import streamlit as st2

from market_engine import app as agent_app, ask_copilot, CostTracker, estimate_cost

st.set_page_config(page_title="Triangulate AI", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background-color: #0B1120; color: #E2E8F0; font-family: 'Inter', sans-serif; }
[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.stButton>button {
    background-color: #2563EB; color: white; border-radius: 6px; border: none; transition: all 0.3s ease;
}
.stButton>button:hover { background-color: #1D4ED8; border-color: #60A5FA; box-shadow: 0 0 10px rgba(37, 99, 235, 0.5); }
.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] { font-weight: 500; color: #94A3B8; padding-bottom: 10px; }
.stTabs [aria-selected="true"] { color: #60A5FA !important; border-bottom-color: #60A5FA !important; }
.trace-line { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94A3B8; padding: 2px 0; }
@keyframes agent-pulse {
    0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
    70% { box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); }
    100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- AGENT PIPELINE VISUAL ---
AGENT_META = {
    "planner":          {"icon": "🧭", "persona": "Dispatch",  "label": "Planner"},
    "market_analyst":   {"icon": "🔍", "persona": "Scout",     "label": "Market Analyst"},
    "data_analyst":     {"icon": "📊", "persona": "Ledger",    "label": "Data Analyst"},
    "research_analyst": {"icon": "🌐", "persona": "Pulse",     "label": "Research Analyst"},
    "synthesizer":      {"icon": "✍️", "persona": "Scribe",    "label": "Synthesizer"},
    "verifier":         {"icon": "⚖️", "persona": "Sentinel",  "label": "Verifier"},
}
PIPELINE_ORDER = ["planner", "market_analyst", "data_analyst", "research_analyst", "synthesizer", "verifier"]

_CARD_STYLES = {
    "pending": {"bg": "rgba(30,41,59,0.35)", "border": "#334155", "text": "#64748B", "pulse": False},
    "active":  {"bg": "rgba(37,99,235,0.15)", "border": "#2563EB", "text": "#60A5FA", "pulse": True},
    "done":    {"bg": "rgba(16,185,129,0.12)", "border": "#10B981", "text": "#34D399", "pulse": False},
    "skipped": {"bg": "rgba(30,41,59,0.15)", "border": "#1E293B", "text": "#475569", "pulse": False},
}


def render_agent_card(node_key: str, status: str, note: str = "") -> str:
    meta = AGENT_META[node_key]
    style = _CARD_STYLES[status]
    animation = "animation: agent-pulse 1.6s infinite;" if style["pulse"] else ""
    check = "✓ " if status == "done" else ""
    note_html = f"<div style='font-size:0.68rem; color:{style['text']}; margin-top:6px; line-height:1.3;'>{html.escape(note)}</div>" if note else ""
    return f"""
    <div style="background:{style['bg']}; border:1.5px solid {style['border']}; border-radius:10px;
                padding:14px 8px; text-align:center; min-height:110px; {animation}">
        <div style="font-size:1.4rem;">{meta['icon']}</div>
        <div style="font-weight:700; color:{style['text']}; font-size:0.95rem; margin-top:2px;">{check}{meta['persona']}</div>
        <div style="font-size:0.68rem; color:#94A3B8;">{meta['label']}</div>
        {note_html}
    </div>
    """

# --- SESSION STATE ---
for key, default in [
    ("app_state", "onboarding"),
    ("report_data", None),
    ("market_findings", None),
    ("data_findings", None),
    ("research_findings", None),
    ("agent_trace", []),
    ("verification", None),
    ("messages", []),
    ("csv_data", None),
    ("pdf_text", ""),
    ("task_prompt", ""),
    ("elapsed_seconds", None),
    ("cost_breakdown", None),
    ("manual_hours", 3.0),
    ("hourly_rate", 45.0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# VIEW 1: ONBOARDING
# ==========================================
if st.session_state.app_state == "onboarding":
    st.markdown("<h1 style='text-align: center; margin-top: 8vh; font-size: 4rem; color: #FFFFFF;'>⚡ Triangulate AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; margin-bottom: 50px; color: #94A3B8;'>Enterprise intelligence, synthesized and self-checked.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 1. Connect Knowledge Base")
        c_pdf, c_csv = st.columns(2)
        with c_pdf:
            uploaded_pdf = st.file_uploader("Strategy / Context (PDF)", type="pdf")
        with c_csv:
            uploaded_csv = st.file_uploader("Raw Database (CSV, optional)", type="csv")

        st.caption(
            "💡 A CSV is optional — the Data Analyst can query the internal SQL "
            "database directly. Upload a CSV only if you want it to analyze your own data too."
        )

        st.divider()
        st.markdown("### 2. Strategic Directive")
        task = st.text_area(
            "What are we looking for?",
            "Analyze the provided inputs to identify key revenue drivers and strategic actions.",
            height=80, label_visibility="collapsed",
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Initialize Engine", type="primary", use_container_width=True):
            st.session_state.pdf_text = (
                "".join(page.extract_text() or "" for page in PyPDF2.PdfReader(uploaded_pdf).pages)
                if uploaded_pdf else "No PDF context provided."
            )
            st.session_state.csv_data = pd.read_csv(uploaded_csv) if uploaded_csv else None
            st.session_state.task_prompt = task
            st.session_state.app_state = "processing"
            st.rerun()

# ==========================================
# VIEW 2: LIVE AGENT GRAPH PROGRESS
# ==========================================
elif st.session_state.app_state == "processing":
    st.markdown("<h2 style='text-align: center; margin-top: 8vh; color: #FFFFFF;'>⚡ Running the Agent Graph</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8; margin-bottom: 30px;'>Watch each specialist agent go to work in real time.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        card_cols = st.columns(len(PIPELINE_ORDER))
        card_placeholders = {}
        agent_status = {k: "pending" for k in PIPELINE_ORDER}
        agent_note = {k: "" for k in PIPELINE_ORDER}

        def render_all():
            for key in PIPELINE_ORDER:
                card_placeholders[key].markdown(
                    render_agent_card(key, agent_status[key], agent_note[key]), unsafe_allow_html=True
                )

        for i, key in enumerate(PIPELINE_ORDER):
            with card_cols[i]:
                card_placeholders[key] = st.empty()
        render_all()

        st.markdown("<br>", unsafe_allow_html=True)
        status_text = st.empty()
        detail_log = st.expander("🔍 View detailed agent log", expanded=False)

        csv_json = st.session_state.csv_data.to_json(orient="records") if st.session_state.csv_data is not None else "[]"
        inputs = {
            "task": st.session_state.task_prompt,
            "pdf_text": st.session_state.pdf_text,
            "csv_json": csv_json,
        }
        tracker = CostTracker()
        config = {
            "configurable": {"thread_id": "workspace-session"},
            "callbacks": [tracker],
        }

        agent_status["planner"] = "active"
        render_all()
        status_text.markdown("<p style='text-align:center; color:#60A5FA;'>🧭 Dispatch is routing the request...</p>", unsafe_allow_html=True)

        start_time = time.time()
        try:
            for event in agent_app.stream(inputs, config):
                for node_name, node_output in event.items():
                    for line in node_output.get("agent_trace", []):
                        detail_log.markdown(f"<div class='trace-line'>{html.escape(line)}</div>", unsafe_allow_html=True)

                    if node_name == "planner":
                        agent_status["planner"] = "done"
                        plan = node_output.get("plan", [])
                        for specialist in ["market_analyst", "data_analyst", "research_analyst"]:
                            agent_status[specialist] = "active" if specialist in plan else "skipped"
                        status_text.markdown("<p style='text-align:center; color:#60A5FA;'>Specialist agents are gathering evidence...</p>", unsafe_allow_html=True)

                    elif node_name == "market_analyst":
                        st.session_state.market_findings = node_output["market_findings"]
                        agent_status["market_analyst"] = "done"
                        agent_note["market_analyst"] = f"{len(node_output['market_findings']['cited_points'])} findings"

                    elif node_name == "data_analyst":
                        st.session_state.data_findings = node_output["data_findings"]
                        agent_status["data_analyst"] = "done"
                        agent_note["data_analyst"] = f"{len(node_output['data_findings']['computed_facts'])} facts"

                    elif node_name == "research_analyst":
                        st.session_state.research_findings = node_output["research_findings"]
                        agent_status["research_analyst"] = "done"
                        agent_note["research_analyst"] = f"{len(node_output['research_findings']['cited_points'])} findings"

                    elif node_name == "synthesizer":
                        agent_status["synthesizer"] = "done"
                        agent_status["verifier"] = "active"
                        status_text.markdown("<p style='text-align:center; color:#60A5FA;'>⚖️ Sentinel is fact-checking the draft...</p>", unsafe_allow_html=True)

                    elif node_name == "verifier":
                        st.session_state.verification = node_output["verification"]
                        agent_status["verifier"] = "done" if node_output["verification"]["passed"] else "active"
                        agent_note["verifier"] = "passed" if node_output["verification"]["passed"] else "revising..."

                    elif node_name == "bump_revision":
                        agent_status["synthesizer"] = "active"
                        agent_status["verifier"] = "pending"
                        status_text.markdown("<p style='text-align:center; color:#FBBF24;'>↩️ Sentinel sent it back — Scribe is revising...</p>", unsafe_allow_html=True)

                    elif node_name == "finalize":
                        st.session_state.report_data = node_output["report_data"]

                    render_all()

            status_text.markdown("<p style='text-align:center; color:#34D399;'>✅ Analysis complete</p>", unsafe_allow_html=True)
            st.session_state.elapsed_seconds = time.time() - start_time
            st.session_state.cost_breakdown = estimate_cost(tracker)
            time.sleep(0.8)
            st.session_state.app_state = "dashboard"
            st.rerun()
        except Exception as e:
            status_text.markdown(f"<p style='text-align:center; color:#F87171;'>❌ Engine error: {html.escape(str(e))}</p>", unsafe_allow_html=True)
            if st.button("← Back to upload"):
                st.session_state.app_state = "onboarding"
                st.rerun()

# ==========================================
# VIEW 3: DASHBOARD
# ==========================================
elif st.session_state.app_state == "dashboard":
    head_col1, head_col2 = st.columns([8, 2])
    with head_col1:
        st.markdown("<h3 style='margin: 0; padding: 0;'>⚡ Triangulate Workspace</h3>", unsafe_allow_html=True)
    with head_col2:
        if st.button("Start New Analysis", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.divider()

    main_col, chat_col = st.columns([7, 3], gap="large")

    with main_col:
        df = st.session_state.csv_data
        report = st.session_state.report_data

        verification = st.session_state.verification or {}
        if verification.get("passed"):
            st.success("✅ Verified — every claim in this report is traced to a source excerpt, an executed query, or a live web result.")
        elif verification:
            st.warning(
                f"⚠️ Shipped after max revisions with {len(verification.get('issues', []))} unresolved issue(s) — "
                "see the Grounding & Trust tab before presenting this externally."
            )

        cost = st.session_state.cost_breakdown
        elapsed = st.session_state.elapsed_seconds
        if cost and elapsed:
            st.caption(f"Generated in {elapsed:.1f}s · ${cost['total_cost']:.4f} in API costs · see the Cost & Throughput tab for details")

        st.markdown("<br>", unsafe_allow_html=True)
        metrics = st.columns(3)
        with metrics[0]:
            st.metric("Data Points Processed", f"{len(df) * len(df.columns):,}" if df is not None else "internal DB")
        with metrics[1]:
            if df is not None:
                rev_cols = [c for c in df.columns if any(w in c.lower() for w in ["rev", "sales", "spend"])]
                if rev_cols:
                    st.metric(f"Total {rev_cols[0].title()}", f"${df[rev_cols[0]].sum():,.2f}")
                else:
                    st.metric("Variables Mapped", len(df.columns))
            else:
                st.metric("Variables Mapped", "via SQL")
        with metrics[2]:
            st.metric("Strategic Vectors", len(report.strategic_recommendations))

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📑 Strategic Brief", "🧾 Grounding & Trust", "📈 Visuals", "🎛️ Data Studio", "💰 Cost & Throughput"])

        with tab1:
            st.markdown("#### Executive Summary")
            st.info(report.executive_summary)
            st.markdown("#### Recommended Action Plan")
            for i, rec in enumerate(report.strategic_recommendations):
                st.markdown(f"**{i+1}.** {rec}")
            if report.key_risks:
                st.markdown("#### Key Risks")
                for risk in report.key_risks:
                    st.markdown(f"- {risk}")

        with tab2:
            st.markdown("#### What the report is grounded in")
            st.caption("This is exactly what the Verifier agent checked the draft against before approving it.")

            if st.session_state.market_findings:
                st.markdown("**🔍 Market Analyst findings** (retrieved from the PDF via TF-IDF search)")
                for point in st.session_state.market_findings["cited_points"]:
                    st.markdown(f"- {point}")

            if st.session_state.data_findings:
                st.markdown("**📊 Data Analyst findings** (from SQL / pandas queries it ran itself)")
                for fact in st.session_state.data_findings["computed_facts"]:
                    st.markdown(f"- {fact}")

            if st.session_state.research_findings:
                st.markdown("**🌐 Research Analyst findings** (from live web search)")
                for point in st.session_state.research_findings["cited_points"]:
                    st.markdown(f"- {point}")

            if report.citations:
                st.markdown("**📎 Report citations**")
                for c in report.citations:
                    st.caption(f"↳ {c}")

            st.markdown("**🧑‍⚖️ Verifier verdict**")
            if verification.get("passed"):
                st.success("Passed on first pass — no unsupported claims found.")
            else:
                for issue in verification.get("issues", []):
                    st.markdown(f"- ⚠️ {issue}")

        with tab3:
            if df is not None:
                all_num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
                smart_num_cols = [c for c in all_num_cols if "id" not in c.lower() and "zip" not in c.lower()]
                categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
                if smart_num_cols and categorical_cols:
                    primary_metric = next((c for c in smart_num_cols if any(w in c.lower() for w in ["rev", "sales", "spend"])), smart_num_cols[0])
                    primary_cat = categorical_cols[0]
                    fig = px.bar(df, x=primary_cat, y=primary_metric, color=primary_cat, title=f"{primary_metric.replace('_', ' ').title()} Breakdown")
                    fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Chart engine requires both categorical and descriptive numeric data.")
            else:
                st.info("No CSV uploaded for this run — the Data Analyst used the internal SQL database instead. Check the Grounding & Trust tab for what it found.")

        with tab4:
            if df is not None:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No CSV uploaded for this run.")

        with tab5:
            if cost and elapsed:
                st.markdown("#### What this run actually cost")
                st.caption(
                    "Real measured numbers from this run — not estimates. Pricing is Groq's and "
                    "Tavily's published per-token/per-search rates as of early 2025; check "
                    "groq.com/pricing and tavily.com/pricing for current rates."
                )
                fact_cols = st.columns(4)
                with fact_cols[0]:
                    st.metric("Time", f"{elapsed:.1f}s")
                with fact_cols[1]:
                    st.metric("Total cost", f"${cost['total_cost']:.4f}")
                with fact_cols[2]:
                    st.metric("Tokens", f"{cost['input_tokens'] + cost['output_tokens']:,}")
                with fact_cols[3]:
                    st.metric("LLM + web calls", f"{cost['llm_calls']} + {cost['tavily_searches']}")

                st.markdown(
                    f"- **Groq (LLM) cost:** ${cost['groq_cost']:.4f} — {cost['input_tokens']:,} input / "
                    f"{cost['output_tokens']:,} output tokens across {cost['llm_calls']} calls\n"
                    f"- **Tavily (web search) cost:** ${cost['tavily_cost']:.4f} — {cost['tavily_searches']} searches"
                )

                st.divider()
                st.markdown("#### What this means at scale")
                st.caption(
                    "This isn't a claim that AI replaces analyst judgment — verification, framing, and "
                    "decisions still need a human (see Grounding & Trust). What it changes is where an "
                    "analyst's time goes: from manually gathering scattered data, to reviewing and acting "
                    "on an already-synthesized, cited draft."
                )

                n_scenarios = st.number_input(
                    "If you needed to run this across N products, regions, or competitors:",
                    min_value=1, max_value=1000, value=10, step=1,
                )
                scaled_cost = cost["total_cost"] * n_scenarios
                scaled_time_minutes = (elapsed * n_scenarios) / 60

                st.session_state.manual_hours = st.number_input(
                    "Rough estimate: analyst-hours to manually research ONE of these",
                    min_value=0.1, max_value=40.0, value=st.session_state.manual_hours, step=0.5,
                    help="This is your own estimate to compare against — not a benchmark. Adjust it to whatever's realistic for this kind of work.",
                )
                st.session_state.hourly_rate = st.number_input(
                    "Analyst hourly rate ($)", min_value=1.0, max_value=500.0,
                    value=st.session_state.hourly_rate, step=5.0,
                )
                manual_hours_total = st.session_state.manual_hours * n_scenarios
                manual_cost_total = manual_hours_total * st.session_state.hourly_rate

                scenario_cols = st.columns(2)
                with scenario_cols[0]:
                    st.markdown("**This pipeline**")
                    st.markdown(f"- ${scaled_cost:,.2f} total cost\n- ~{scaled_time_minutes:.1f} minutes (sequential)")
                with scenario_cols[1]:
                    st.markdown("**Manual research at your estimate**")
                    st.markdown(f"- ~{manual_hours_total:,.1f} analyst-hours\n- ~${manual_cost_total:,.0f} at ${st.session_state.hourly_rate:.0f}/hr")

                st.caption(
                    "Read this as: the pipeline removes the data-gathering bottleneck across many items, "
                    "freeing analyst hours for the judgment calls an AI shouldn't make alone."
                )
            else:
                st.info("Cost data will appear here after a run completes.")

    with chat_col:
        st.markdown("<h4 style='margin-bottom: 5px;'>💬 Strategic Copilot</h4>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.85rem; color: #94A3B8; margin-bottom: 15px;'>Ask questions grounded in the verified report.</p>", unsafe_allow_html=True)

        chat_container = st.container(height=500, border=False)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("*Context loaded. What should we drill down into?*")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        user_prompt = st.session_state.messages[-1]["content"]
                        response = ask_copilot(user_prompt, st.session_state.report_data, st.session_state.messages)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()