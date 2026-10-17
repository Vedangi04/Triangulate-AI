# Triangulate AI

A multi-agent market intelligence assistant. Ask a market question and three specialist agents each investigate from a different data source. A synthesizer then writes a cited report, and a verifier checks it against the evidence before it ships.

## How it works

```
Planner --> { Market Analyst, Data Analyst, Research Analyst } --> Synthesizer
                                                                       |
                                                                       v
                                                                   Verifier
                                                                   /       \
                                                       (issues found)     (passed)
                                                             |                |
                                                   back to Synthesizer       END
                                                      (max 2 retries)
```

| Agent | Data source | Method |
|---|---|---|
| Market Analyst | Uploaded PDF (internal docs) | TF-IDF retrieval |
| Data Analyst | `company_data.db` (SQLite) and an optional uploaded CSV | ReAct agent with SQL and pandas tools |
| Research Analyst | Live web | ReAct agent with Tavily search |

The pipeline is built with [LangGraph](https://github.com/langchain-ai/langgraph) and the UI with [Streamlit](https://streamlit.io). The LLM is served through [Groq](https://groq.com).

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/Vedangi04/Triangulate-AI.git
cd Triangulate-AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

Optionally, recreate the sample sales database:

```bash
python setup_db.py
```

## Run

```bash
streamlit run app.py
```

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI and live agent pipeline view |
| `market_engine.py` | LangGraph pipeline: planner, analysts, synthesizer, verifier, cost tracking |
| `tools.py` | Text chunking, TF-IDF retrieval, dataframe profiling and pandas query tools |
| `sql_tool.py` | Read-only SQL tool over `company_data.db` |
| `pdf_tool.py` | PDF reader for the `knowledge/` folder |
| `setup_db.py` | Creates `company_data.db` with mock semiconductor sales data |
| `knowledge/` | Sample internal strategy document |

## Notes

- `company_data.db` contains mock data for demonstration only.
- Never commit your `.env` file. It is already in `.gitignore`.
