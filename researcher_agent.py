import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 1. Initialize the Free Brain (Using Llama 3.3)
# This model is incredibly fast and smart for research tasks.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# 2. Set up the Search Tool
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# 3. Create the Agent
agent_executor = create_react_agent(llm, tools)

# 4. Run a Market Intelligence Query
print("--- 🧠 Free Researcher Agent is Searching ---")
input_data = {"messages": [("user", "Summarize the latest 2026 updates on NVIDIA's competition with Huawei in AI chips.")]}

try:
    for chunk in agent_executor.stream(input_data, stream_mode="values"):
        message = chunk["messages"][-1]
        if hasattr(message, "content"):
            print(f"\n[Agent]: {message.content[:200]}...") # Printing snippets as it thinks
    
    print("\n✅ DONE!")
except Exception as e:
    print(f"❌ Error: {e}")