import json
from cat.log import log

from langchain.output_parsers import CommaSeparatedListOutputParser

def clean_langchain_query(query):
    if query.startswith("SQLQuery: "):
        query = query.split(": ")[1]
    query = query.replace("\n", " ")
    query = query.replace("```", "")
    if "sql " in query:
        query = query.split("sql ")[1]
    return query.strip()

def extract_columns_from_query(llm, sql_statement):
    query_lower = sql_statement.lower()
    query_types = [
        "select",
        "show",
        "describe",
        "explain",
        "with"
    ]
    
    # Check if the query is a SELECT, SHOW, DESCRIBE, EXPLAIN, or WITH query, or if it contains a RETURNING clause
    try:
        if any(query_lower.startswith(qt) for qt in query_types) or " returning " in query_lower:
            llm_output = llm.invoke(f"""You extract column names from SQL statements. If no proper column names can be extracted (e.g. due to use of wildcards), reply with "" to skip this step.
Given the following SQL statement, extract the column names and return them as a comma-separated list: {sql_statement}""").content

            if llm_output:
                return CommaSeparatedListOutputParser().parse(llm_output)
            else:
                return []
        else:
            # If this query does not return any columns, return an empty list without invoking the LLM
            return []
    except Exception as e:
        log.error(f"Error extracting columns from query: {e}")
        return []
