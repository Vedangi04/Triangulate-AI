"""
market_engine.py
-----------------
The brain of Triangulate AI.

Architecture (see README.md for the diagram):

    Planner --> {Market Analyst, Data Analyst, Research Analyst} --> Synthesizer
                                                                          |
                                                                          v
                                                                      Verifier
                                                                      /       \\
                                                          (issues found)     (passed)
                                                                |                |
                                                          back to Synthesizer   END
                                                          (max 2 retries)

Three specialist agents, three different tools, three different data sources:
- Market Analyst: TF-IDF retrieval over an uploaded PDF (internal context docs).
- Data Analyst: a real ReAct tool-calling agent (langgraph's create_react_agent)
  with access to a SQL tool (queries company_data.db) and a pandas tool (queries
  an uploaded CSV, if any) — it decides which tool(s) to use.
- Research Analyst: a ReAct agent with a live web search tool (Tavily) — gives
  the report real external market context, not just what the user uploaded.

All three feed grounded, cited findings into the Synthesizer, and the Verifier
checks the draft against those findings before it's allowed to ship.
"""

from __future__ import annotations

import os
import json
import io
import operator
from typing import Annotated, List, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from tools import chunk_text, TfidfRetriever, dataframe_profile, make_dataframe_tool
from sql_tool import query_sales_database

load_dotenv()

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_REVISIONS = 2
HAS_TAVILY = bool(os.environ.get("TAVILY_API_KEY"))

if HAS_TAVILY:
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=3)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.2,
)

# ReAct agents use temperature=0 — we want reliable tool-use, not creativity.
tool_calling_llm = ChatGroq(
    model=MODEL_NAME,
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
)


# ---------------------------------------------------------------------------
# Cost tracking — powers the ROI calculator in the UI.
#
# NOTE on pricing: these are approximate published rates as of early 2025 for
# llama-3.3-70b-versatile on Groq, and Tavily's pay-as-you-go rate beyond the
# free tier. Prices change — check groq.com/pricing and tavily.com/pricing
# for current numbers. Treat these as editable estimates, not audited figures.
# ---------------------------------------------------------------------------

GROQ_INPUT_PRICE_PER_MILLION = 0.59   # USD per 1M input tokens
GROQ_OUTPUT_PRICE_PER_MILLION = 0.79  # USD per 1M output tokens
TAVILY_PRICE_PER_SEARCH = 0.008       # USD per search beyond free tier


class CostTracker(BaseCallbackHandler):
    """
    A LangChain callback that accumulates real token usage and tool-call
    counts across an entire graph run (including nested ReAct agents),
    so we can compute an honest cost figure instead of guessing.
    """

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.tool_calls: dict = {}

    def on_llm_end(self, response, **kwargs):
        self.llm_calls += 1
        for generation_list in response.generations:
            for generation in generation_list:
                msg = getattr(generation, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg else None
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0) or 0
                    self.output_tokens += usage.get("output_tokens", 0) or 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown_tool")
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1


def estimate_cost(tracker: "CostTracker") -> dict:
    groq_cost = (
        (tracker.input_tokens / 1_000_000) * GROQ_INPUT_PRICE_PER_MILLION
        + (tracker.output_tokens / 1_000_000) * GROQ_OUTPUT_PRICE_PER_MILLION
    )
    tavily_searches = tracker.tool_calls.get("tavily_search", 0)
    tavily_cost = tavily_searches * TAVILY_PRICE_PER_SEARCH
    return {
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "llm_calls": tracker.llm_calls,
        "tavily_searches": tavily_searches,
        "groq_cost": groq_cost,
        "tavily_cost": tavily_cost,
        "total_cost": groq_cost + tavily_cost,
    }


# ---------------------------------------------------------------------------
# Structured output schemas (Pydantic) — every agent that writes report
# content is forced through one of these, so the app never has to parse
# free-form text.
# ---------------------------------------------------------------------------

class MarketFindings(BaseModel):
    summary: str = Field(description="What the market/context PDF tells us, in 3-5 sentences.")
    cited_points: List[str] = Field(
        default_factory=list,
        description="Specific claims pulled from the PDF, each phrased as 'claim (source: chunk N)'.",
    )


class DataFindings(BaseModel):
    summary: str = Field(description="What the internal data shows, in 3-5 sentences.")
    computed_facts: List[str] = Field(
        default_factory=list,
        description="Specific facts computed via SQL or pandas, each phrased as 'fact (source: <query used>)'.",
    )


class ResearchFindings(BaseModel):
    summary: str = Field(description="What live web research turned up relevant to the task, in 3-5 sentences.")
    cited_points: List[str] = Field(
        default_factory=list,
        description="Specific external facts, each phrased as 'claim (source: <url>)'.",
    )


class StrategicReport(BaseModel):
    executive_summary: str
    strategic_recommendations: List[str]
    key_risks: List[str] = Field(default_factory=list)
    citations: List[str] = Field(
        default_factory=list,
        description="Every factual claim above traced back to its source (pdf chunk, SQL/pandas query, or URL).",
    )


class VerificationResult(BaseModel):
    passed: bool
    issues: List[str] = Field(
        default_factory=list,
        description="Specific unsupported or contradicted claims found in the draft, with why they fail.",
    )
    revision_instructions: str = Field(
        default="", description="Concrete instructions for the Synthesizer to fix the issues."
    )


class PlanDecision(BaseModel):
    use_data_analyst: bool = Field(
        description="True only if the internal SQL sales database (GPU hardware revenue: "
        "H100/A100/Blackwell units, revenue, quarter) is actually relevant to answering the task. "
        "False for tasks unrelated to this company's product sales, e.g. resumes, unrelated "
        "documents, or topics with no connection to GPU sales."
    )
    use_research_analyst: bool = Field(
        description="True if live web research would meaningfully help answer the task."
    )
    reasoning: str = Field(description="One sentence explaining the routing decision.")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    task: str
    pdf_text: str
    csv_json: str

    plan: List[str]                       # which specialists to run, decided by the Planner
    market_findings: Optional[dict]
    data_findings: Optional[dict]
    research_findings: Optional[dict]

    draft_report: Optional[dict]
    verification: Optional[dict]
    revision_count: int

    report_data: StrategicReport          # final validated output the UI reads

    # Annotated with operator.add because up to three specialist nodes can run
    # in the SAME step (fan-out from the Planner) and all append to this log —
    # LangGraph needs to know to concatenate rather than overwrite.
    agent_trace: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# Helper: pull the final text answer out of a create_react_agent transcript
# ---------------------------------------------------------------------------

def _invoke_structured(structured_llm, prompt: str, config: RunnableConfig, retries: int = 1):
    """
    Groq's tool-calling models occasionally emit malformed types in their
    function-call output — e.g. "false" (a string) instead of false (a JSON
    boolean) — which the API hard-rejects with a 400 error before we ever
    see a response to fix up ourselves. This isn't specific to our schema;
    it's a known flakiness pattern in tool-calling LLMs generally. Retry
    once with an explicit type reminder before giving up.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return structured_llm.invoke(prompt, config=config)
        except Exception as e:
            last_error = e
            prompt = prompt + (
                "\n\nIMPORTANT: return exact JSON types per the schema — booleans must be "
                "true/false (not \"true\"/\"false\" strings), numbers must be numeric, not quoted."
            )
    raise last_error


def _extract_react_transcript(response: dict) -> str:
    """
    create_react_agent returns {"messages": [...]}. We want a compact record
    of what tools were called with what results, plus the agent's final
    answer, so we can hand it to a structured-output call afterward.
    """
    lines = []
    for m in response["messages"]:
        role = m.__class__.__name__
        content = getattr(m, "content", "")
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                lines.append(f"[tool call] {tc['name']}({tc['args']})")
        if content:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: Planner
# ---------------------------------------------------------------------------

def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    plan = []
    has_pdf = bool(state.get("pdf_text")) and state["pdf_text"].strip().lower() != "no pdf context provided."
    has_csv = bool(state.get("csv_json")) and state["csv_json"] not in ("[]", "", None)

    if has_pdf:
        plan.append("market_analyst")

    if has_csv:
        # The user explicitly uploaded their own data — always worth analyzing,
        # regardless of whether the internal demo SQL database is relevant.
        plan.append("data_analyst")
        decision = None
    else:
        structured_llm = llm.with_structured_output(PlanDecision)
        try:
            decision: PlanDecision = _invoke_structured(
                structured_llm,
                "Decide which specialist agents are actually relevant to this task. The internal "
                "SQL database ONLY contains GPU hardware sales data (H100, A100, and Blackwell B200 "
                "units sold / revenue / quarter) for one semiconductor company. It is NOT relevant to "
                "unrelated tasks — e.g. analyzing a resume, an unrelated document, or any topic with no "
                "connection to this company's GPU sales. Don't force a data source into the plan just "
                "because it exists.\n\n"
                f"TASK: {state['task']}\n\n"
                f"UPLOADED DOCUMENT (first 400 chars, may be empty): {state.get('pdf_text', '')[:400]}",
                config,
            )
        except Exception as e:
            # Even after a retry, the model's tool call was malformed — don't
            # crash the whole run over a routing decision. Fail safe: skip
            # both optional specialists rather than guess wrong.
            decision = PlanDecision(
                use_data_analyst=False,
                use_research_analyst=False,
                reasoning=f"Routing model call failed twice ({type(e).__name__}) — defaulting to Market Analyst only.",
            )
        if decision.use_data_analyst:
            plan.append("data_analyst")

    use_research = (decision.use_research_analyst if decision else True)
    if use_research and HAS_TAVILY:
        plan.append("research_analyst")

    reasoning = decision.reasoning if decision else "CSV uploaded — analyzing user's own data."
    trace = [f"🧭 Dispatch (Planner): routing to {plan} — {reasoning}"]
    if use_research and not HAS_TAVILY:
        trace.append("ℹ️ Planner: web research would help but TAVILY_API_KEY not set — skipping.")
    return {"plan": plan, "agent_trace": trace, "revision_count": 0}


def route_after_planner(state: AgentState) -> list:
    return state.get("plan") or ["synthesizer"]


# ---------------------------------------------------------------------------
# Node: Market Analyst (RAG over the PDF)
# ---------------------------------------------------------------------------

def market_analyst_node(state: AgentState, config: RunnableConfig) -> dict:
    chunks = chunk_text(state.get("pdf_text", ""))
    retriever = TfidfRetriever(chunks)
    relevant = retriever.retrieve(state["task"], k=5)
    context = "\n\n".join(f"[chunk {c.chunk_id}] {c.text}" for c in relevant) or "No relevant chunks found."

    structured_llm = llm.with_structured_output(MarketFindings)
    findings: MarketFindings = _invoke_structured(
        structured_llm,
        "You are a market research analyst. Using ONLY the retrieved context below, "
        "extract findings relevant to the task. Every point in `cited_points` MUST "
        "reference the chunk number, e.g. '... (source: chunk 3)'. Do not invent "
        "information that isn't in the context.\n\n"
        f"TASK: {state['task']}\n\nRETRIEVED CONTEXT:\n{context}",
        config,
    )

    trace = [f"🔍 Scout (Market Analyst): retrieved {len(relevant)} chunk(s), extracted {len(findings.cited_points)} cited findings"]
    return {"market_findings": findings.model_dump(), "agent_trace": trace}


# ---------------------------------------------------------------------------
# Node: Data Analyst — a real ReAct agent with SQL + (optionally) pandas tools
# ---------------------------------------------------------------------------

DATA_ANALYST_SYSTEM_PROMPT = """You are a Senior Data Analyst. You have access to:
1. 'query_sales_database': queries the internal SQL database. Table 'sales' has
   columns: id, product_name, units_sold, revenue, quarter. product_name values
   have suffixes like 'GPU' — always use LIKE with wildcards, e.g.
   SELECT * FROM sales WHERE product_name LIKE '%H100%'. Quarters are formatted
   like '2025-Q4'.
2. 'query_uploaded_csv' (if available): queries the user's uploaded CSV via pandas.

Use whichever tool(s) are relevant to the task. Run at least one concrete query —
do not answer from assumptions. Report only what the queries actually returned."""


def data_analyst_node(state: AgentState, config: RunnableConfig) -> dict:
    tools = [query_sales_database]
    has_csv = bool(state.get("csv_json")) and state["csv_json"] not in ("[]", "", None)
    if has_csv:
        df = pd.read_json(io.StringIO(state["csv_json"]), orient="records")
        tools.append(make_dataframe_tool(df))

    agent = create_react_agent(tool_calling_llm, tools, prompt=DATA_ANALYST_SYSTEM_PROMPT)
    response = agent.invoke({"messages": [("user", state["task"])]}, config=config)
    transcript = _extract_react_transcript(response)

    structured_llm = llm.with_structured_output(DataFindings)
    findings: DataFindings = _invoke_structured(
        structured_llm,
        "You are summarizing a data analyst agent's tool-calling transcript. Using "
        "ONLY the real tool call results below (never invent numbers), extract facts "
        "relevant to the task. Every item in `computed_facts` must reference which "
        "query produced it.\n\n"
        f"TASK: {state['task']}\n\nTRANSCRIPT:\n{transcript}",
        config,
    )

    trace = [f"📊 Ledger (Data Analyst): used {[t.name for t in tools]}, extracted {len(findings.computed_facts)} computed facts"]
    return {"data_findings": findings.model_dump(), "agent_trace": trace}


# ---------------------------------------------------------------------------
# Node: Research Analyst — a ReAct agent with live web search (Tavily)
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """You are a Market Research Analyst with access to live web
search ('tavily_search'). Search for current, specific, relevant news for the task —
prefer precise queries (e.g. "NVIDIA Blackwell GPU" rather than just "Blackwell", which
collides with unrelated people/places). Report only what search results actually say."""


def research_analyst_node(state: AgentState, config: RunnableConfig) -> dict:
    agent = create_react_agent(tool_calling_llm, [search_tool], prompt=RESEARCH_SYSTEM_PROMPT)
    response = agent.invoke({"messages": [("user", state["task"])]}, config=config)
    transcript = _extract_react_transcript(response)

    structured_llm = llm.with_structured_output(ResearchFindings)
    findings: ResearchFindings = _invoke_structured(
        structured_llm,
        "You are summarizing a research agent's web-search transcript. Using ONLY the "
        "search results below, extract findings relevant to the task. Every item in "
        "`cited_points` must reference the source URL. If the search results are "
        "irrelevant to the task, say so in `summary` and leave `cited_points` empty "
        "rather than forcing a connection.\n\n"
        f"TASK: {state['task']}\n\nTRANSCRIPT:\n{transcript}",
        config,
    )

    trace = [f"🌐 Pulse (Research Analyst): searched the web, extracted {len(findings.cited_points)} cited findings"]
    return {"research_findings": findings.model_dump(), "agent_trace": trace}


# ---------------------------------------------------------------------------
# Node: Synthesizer
# ---------------------------------------------------------------------------

def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    market = state.get("market_findings")
    data = state.get("data_findings")
    research = state.get("research_findings")

    revision_note = ""
    if state.get("verification") and not state["verification"]["passed"]:
        revision_note = (
            "\n\nIMPORTANT — this is a REVISION. The previous draft had issues:\n"
            + "\n".join(f"- {i}" for i in state["verification"]["issues"])
            + f"\n\nRevision instructions: {state['verification']['revision_instructions']}"
            + "\nFix these specifically. Do not repeat the same unsupported claims."
        )

    context_block = ""
    if market:
        context_block += f"\nMARKET ANALYST FINDINGS (from PDF):\n{market['summary']}\n" + "\n".join(f"- {p}" for p in market["cited_points"])
    if data:
        context_block += f"\n\nDATA ANALYST FINDINGS (from SQL/CSV):\n{data['summary']}\n" + "\n".join(f"- {f}" for f in data["computed_facts"])
    if research:
        context_block += f"\n\nRESEARCH ANALYST FINDINGS (from live web search):\n{research['summary']}\n" + "\n".join(f"- {p}" for p in research["cited_points"])
    if not context_block:
        context_block = "\nNo source documents were provided — be explicit that recommendations are general best practice, not data-derived."

    structured_llm = llm.with_structured_output(StrategicReport)
    report: StrategicReport = _invoke_structured(
        structured_llm,
        "You are a strategy consultant writing an executive report. Base EVERY factual "
        "claim strictly on the findings below — never invent numbers, sources, or events. "
        "If any findings set below is NOT actually relevant to the task (e.g. it discusses "
        "an unrelated topic), IGNORE that findings set entirely rather than forcing a "
        "connection — do not let irrelevant data hijack the report's focus. "
        "Populate `citations` by copying the source references from the findings that "
        "back up your claims.\n"
        f"\nTASK: {state['task']}\n{context_block}{revision_note}",
        config,
    )

    trace = ["✍️ Scribe (Synthesizer): drafted strategic report" + (" (revision)" if revision_note else "")]
    return {"draft_report": report.model_dump(), "agent_trace": trace}


# ---------------------------------------------------------------------------
# Node: Verifier — the trust layer
# ---------------------------------------------------------------------------

def verifier_node(state: AgentState, config: RunnableConfig) -> dict:
    draft = state["draft_report"]
    market = state.get("market_findings") or {}
    data = state.get("data_findings") or {}
    research = state.get("research_findings") or {}

    source_facts = (
        "MARKET CITED POINTS:\n" + "\n".join(market.get("cited_points", [])) +
        "\n\nDATA COMPUTED FACTS:\n" + "\n".join(data.get("computed_facts", [])) +
        "\n\nRESEARCH CITED POINTS:\n" + "\n".join(research.get("cited_points", []))
    )

    structured_llm = llm.with_structured_output(VerificationResult)
    result: VerificationResult = _invoke_structured(
        structured_llm,
        "You are a fact-checking critic. Compare the DRAFT REPORT against the SOURCE "
        "FACTS it was supposed to be grounded in. Flag any claim in the draft that is "
        "NOT supported by the source facts (a hallucination), any number that doesn't "
        "match, or any citation referencing a source that doesn't exist. If a report "
        "has no source facts at all, pass it as long as it doesn't falsely claim to be "
        "data-derived.\n\n"
        f"SOURCE FACTS:\n{source_facts}\n\nDRAFT REPORT:\n{json.dumps(draft, indent=2)}",
        config,
    )

    trace = [
        "⚖️ Sentinel (Verifier): PASSED" if result.passed
        else f"⚖️ Sentinel (Verifier): found {len(result.issues)} issue(s) — sending back for revision"
    ]
    return {"verification": result.model_dump(), "agent_trace": trace}


def route_after_verification(state: AgentState) -> str:
    verification = state.get("verification", {})
    revision_count = state.get("revision_count", 0)
    if verification.get("passed") or revision_count >= MAX_REVISIONS:
        return "finalize"
    return "revise"


def bump_revision_node(state: AgentState) -> dict:
    return {"revision_count": state.get("revision_count", 0) + 1}


def finalize_node(state: AgentState) -> dict:
    report = StrategicReport(**state["draft_report"])
    verification = state.get("verification", {})
    trace = ["🏁 Finalized report"]
    if not verification.get("passed", True):
        trace.append("ℹ️ Note: shipped after max revisions — remaining issues logged for the user.")
    return {"report_data": report, "agent_trace": trace}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("market_analyst", market_analyst_node)
workflow.add_node("data_analyst", data_analyst_node)
workflow.add_node("research_analyst", research_analyst_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("bump_revision", bump_revision_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "market_analyst": "market_analyst",
        "data_analyst": "data_analyst",
        "research_analyst": "research_analyst",
        "synthesizer": "synthesizer",
    },
)

workflow.add_edge("market_analyst", "synthesizer")
workflow.add_edge("data_analyst", "synthesizer")
workflow.add_edge("research_analyst", "synthesizer")

workflow.add_edge("synthesizer", "verifier")
workflow.add_conditional_edges(
    "verifier",
    route_after_verification,
    {"finalize": "finalize", "revise": "bump_revision"},
)
workflow.add_edge("bump_revision", "synthesizer")
workflow.add_edge("finalize", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Copilot: lightweight, memory-aware chat used by the Workspace page.
# ---------------------------------------------------------------------------

def ask_copilot(question: str, report: StrategicReport, chat_history: list) -> str:
    history_block = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in chat_history[-6:])
    prompt = (
        "You are the Strategic Copilot for Triangulate AI. Answer the user's question "
        "using the verified report below as ground truth. If the question asks about "
        "something outside the report, say so rather than inventing an answer.\n\n"
        f"VERIFIED REPORT:\nExecutive Summary: {report.executive_summary}\n"
        f"Recommendations: {report.strategic_recommendations}\n"
        f"Risks: {report.key_risks}\n\n"
        f"RECENT CONVERSATION:\n{history_block}\n\nUSER QUESTION: {question}"
    )
    return llm.invoke(prompt).content