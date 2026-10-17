import operator
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END

# 1. Define the 'Notebook' (State)
class AgentState(TypedDict):
    query: str
    research_notes: str
    financial_data: str
    final_report: str

# 2. Define Node 1: The Researcher
def researcher_node(state: AgentState):
    print("--- 🔍 Researcher is hitting the web ---")
    # Here we would call Tavily (simplified for now)
    return {"research_notes": "Blackwell GPUs are seeing massive deployment by Akamai in March 2026."}

# 3. Define Node 2: The Financial Analyst
def analyst_node(state: AgentState):
    print("--- 📊 Analyst is querying SQL ---")
    # Here we would call our SQL Tool
    return {"final_report": f"Based on news ({state['research_notes']}), our H100 sales of $186M in Q4 look strong."}

# 4. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()

# 5. Run the Graph
print("--- 🚀 Starting Multi-Agent Workflow ---")
for output in app.stream({"query": "Blackwell vs H100"}):
    print(output)