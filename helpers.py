import json

def clean_langchain_query(query):
    if query.startswith("SQLQuery: "):
        query = query.split(": ")[1]
    query = query.replace("\n", " ")
    query = query.replace("```", "")
    if "sql " in query:
        query = query.split("sql ")[1]
    return query.strip()

def extract_columns_from_query(llm, query):
    query_lower = query.lower()
    query_types = [
        "select",
        "show",
        "describe",
        "explain",
        "with"
    ]
    
    # Check if the query is a SELECT, SHOW, DESCRIBE, EXPLAIN, or WITH query, or if it contains a RETURNING clause
    if any(query_lower.startswith(qt) for qt in query_types) or " returning " in query_lower:
        columns_json = llm.invoke(f"Extract the result columns from the SQL query and return a JSON list of strings: {query}. If not applicable, reply with '[]'.").content
        return json.loads(columns_json.replace("\n", "").replace("```json", "").replace("```", ""))
    else:
        # If this query does not return any columns, return an empty list without invoking the LLM
        return []
