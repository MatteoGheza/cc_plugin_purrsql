from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel
import json

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

EXAMPLE_DB_URL = "sqlite:///cat/plugins/purrsql/example.db"

class PurrSQLSettings(BaseModel):
    db_url: str = EXAMPLE_DB_URL

@plugin
def settings_model():
    return PurrSQLSettings

@hook
def agent_prompt_prefix(prefix, cat):
    return """You are a DB client. You reply in a complete and precise way to user questions.
You can query a SQLite database from an ordering system.
When suited, provide the data in a markdown table format, with the first row being the key names, else provide the data in a human readable format.
"""

db = None

@hook  # default priority = 1
def after_cat_bootstrap(cat):
    global db
    settings = cat.mad_hatter.get_plugin().load_settings()
    if settings is None or "db_url" not in settings:
        settings = {
            "db_url": EXAMPLE_DB_URL
        }
        cat.mad_hatter.get_plugin().save_settings(settings)
    db = SQLDatabase.from_uri(settings["db_url"])

@tool
def database(tool_input, cat):
    """This plugin should be used when user asks to get, insert, update, filter, delete data from the database.
Data can be ordered or filtered in different ways.
tool_input is a HUMAN FORMATTED STRING, WHICH IS A QUESTION OR COMMAND, NOT SQL QUERY OR ANYTHING ELSE.
The output is a JSON object, with "result" key containing the result of the query and "columns" key containing the column names.
If the query returns error, the "result" key contains the error message string.
PROVIDE THE DATA IN A MARKDOWN TABLE FORMAT, WITH THE FIRST ROW BEING THE KEY NAMES.
Example queries:
- "Show me the list of products"
- "Query orders ordered by added date"
- "Show me the list of products with price less than 10"
Example output:
{
    "result": [(33, 'VINO ROSSO 1L', 6), (16, 'VINO ROSSO 0,5L', 3)]
    "columns": ["id", "name", "price"]
}
{
    "result": "no data found"
}
"""
    global db

    if db is None:
        return "Database is not connected. Please update the settings."

    chain = create_sql_query_chain(cat._llm, db)

    system = """Double check the user's {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- If selecting data from multiple tables, ensure that the join conditions are correct
- If selecting data from multiple tables, ensure that every column name is unique or aliased
- Make sure the table exists and the column names are correct

If there are any of the above mistakes, rewrite the query.
If there are no mistakes, just reproduce the original query with no further commentary.

Output the final SQL query only."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{query}")]
    ).partial(dialect=db.dialect)
    validation_chain = prompt | cat._llm | StrOutputParser()

    full_chain = {"query": chain} | validation_chain

    query = full_chain.invoke({"question": tool_input})

    if query.startswith("SQLQuery: "):
        query = query.split(": ")[1]
    query = query.replace("\n", " ")
    query = query.replace("```", "")
    if "sql " in query:
        query = query.split("sql ")[1]

    try:
        result = str(db.run(query))
        columns_json = cat.llm(f"Extract the result columns from the SQL query and return a JSON list of strings: {query}. If not applicable, reply with '[]'.")
        columns = json.loads(columns_json.replace("\n", "").replace("```", "").replace("json", ""))
        response = {
            "result": result,
            "columns": columns
        }
    except Exception as e:
        response = {
            "result": str(e)
        }
    return json.dumps(response)
