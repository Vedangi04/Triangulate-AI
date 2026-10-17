import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from sql_tool import query_sales_database

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

search_tool = TavilySearch(max_results=3)
tools = [search_tool, query_sales_database]

# UPDATED SYSTEM PROMPT: Giving the AI "Expert Hints"
system_message = """You are a Senior Market Intelligence Analyst. 
You have access to:
1. 'tavily_search': Use this for external news.
2. 'query_sales_database': Use this for internal sales.

CRITICAL INSTRUCTIONS:
- Internal Database: The 'product_name' column usually contains suffixes like 'GPU'. 
  Always use 'LIKE' with wildcards in your SQL. Example: SELECT * FROM sales WHERE product_name LIKE '%H100%'
- Quarters: Our quarters are formatted as '2025-Q3' or '2025-Q4'. Use these exact strings.
- Research: If 'Blackwell' isn't in the top news, search for 'NVIDIA next-gen GPU' or 'NVIDIA Blackwell delay'."""

agent_executor = create_react_agent(llm, tools, prompt=system_message)

print("--- 🧠 Smart Analyst is Analyzing (Fixed Logic) ---")
# I've updated the query to use the format the database expects
query = {"messages": [("user", "Compare the latest Blackwell chip news with our internal H100 sales in 2025-Q4.")]}

try:
    response = agent_executor.invoke(query)
    for m in response["messages"]:
        m.pretty_print()
except Exception as e:
    print(f"❌ Error: {e}")