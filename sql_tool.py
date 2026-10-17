import sqlite3
from langchain_core.tools import tool

@tool
def query_sales_database(query: str) -> str:
    """
    Execute a read-only SQL query against the internal company_data.db.
    Use this to find internal sales, revenue, units sold, and quarter-over-quarter growth.
    The table name is 'sales'. Columns: id, product_name, units_sold, revenue, quarter.
    Only SELECT statements are permitted.
    """
    if not query.strip().upper().startswith("SELECT"):
        return "Error: only SELECT statements are permitted."

    try:
        # Open in read-only mode so even a crafted query can't mutate the DB.
        conn = sqlite3.connect('file:company_data.db?mode=ro', uri=True)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No data found for that query."
        return str(rows)
    except Exception as e:
        return f"Error executing SQL: {e}"