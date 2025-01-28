from cat.mad_hatter.decorators import tool, hook, plugin
from cat.log import log
from cat.experimental.form import CatForm, form
import json
import os

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import sqlparse

from cat.plugins.purrsql.models import HelperLLM, PurrSQLSettings, DBConnectionInfo, EXAMPLE_DB_URL
from cat.plugins.purrsql.helpers import clean_langchain_query, extract_columns_from_query

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
custom_llm = None
enable_query_debugger = True

@hook  # default priority = 1
def after_cat_bootstrap(cat):
    global db, custom_llm, enable_query_debugger
    settings = cat.mad_hatter.get_plugin().load_settings()

    default_settings = {
        "db_url": EXAMPLE_DB_URL,
        "enable_query_debugger": True,
        "helper_llm_api_key": "",
        "helper_llm_model": "",
        "helper_llm": "cat"
    }
    # Check if the settings are missing or incomplete (settings file updated manually or old plugin version)
    if not settings or not all(key in settings for key in default_settings):
        # Merge existing settings with defaults, keeping existing values when present
        settings = {**default_settings, **(settings or {})}
        cat.mad_hatter.get_plugin().save_settings(settings)

    db = SQLDatabase.from_uri(settings["db_url"])

    if settings["helper_llm"] == HelperLLM.llama:
        #TODO
        pass
    elif settings["helper_llm"] == HelperLLM.gemini:
        os.environ["GOOGLE_API_KEY"] = settings["helper_llm_api_key"]
        from langchain_google_genai import ChatGoogleGenerativeAI
        custom_llm = ChatGoogleGenerativeAI(
            model=settings["helper_llm_model"],
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
    
    enable_query_debugger = settings["enable_query_debugger"]

@tool
def database(tool_input, cat):
    """This plugin should be used when user asks to get, insert, update, filter, delete data from the database.
Data can be ordered or filtered in different ways.
tool_input is a HUMAN FORMATTED STRING, WHICH IS A QUESTION OR COMMAND, NOT SQL QUERY OR ANYTHING ELSE.
The command can be multiple requests, separated by "THEN", which will be executed in order.
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
    global db, custom_llm, enable_query_debugger

    if db is None:
        return "Database is not connected. Please update the settings."
    
    # Setup a new Langchain LLM
    llm = custom_llm or cat._llm

    chain = create_sql_query_chain(llm, db)

    system = """You are a SQL query validator and optimizer. Your task is to process every statement (divided by THEN) contained in the input query sequentially.
For each query in the input:
1. Validate the {dialect} query for common mistakes, including:
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

2. If there are any mistakes in a query, rewrite that specific query.
3. If there are no mistakes in a query, reproduce it exactly as is.

Important rules:
- Process and output ALL queries in the original order
- Maintain the original sequence of operations
- Each query must be separated by a semicolon and combined into a single string
- Include a semicolon after the last query
- Do not include any comments in the output
- Do not skip or ignore any queries
"""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{query}")]
    ).partial(dialect=db.dialect)
    validation_chain = prompt | cat._llm | StrOutputParser()

    full_chain = {"query": chain} | validation_chain

    query = full_chain.invoke({"question": tool_input})

    query = clean_langchain_query(query)

    try:
        log.info(query)
        
        if enable_query_debugger:
            cat.send_chat_message(f"""Query SQL eseguita: \n```sql\n{query}\n```""")

        statements = sqlparse.split(query)

        if len(statements) == 0:
            return "No valid query to execute"
        elif len(statements) == 1:
            response = {
                "result": str(db.run(statements[0])),
                "columns": extract_columns_from_query(llm, query)
            }
        else:
            response = []
            for statement in statements:
                # Ignore BEGIN and COMMIT statements if added by Langchain
                if not statement.startswith("BEGIN") and not statement.startswith("COMMIT"):
                    response.append({
                        "result": str(db.run(statement)),
                        "columns": extract_columns_from_query(llm, statement)
                    })
    except Exception as e:
        response = {
            "result": str(e)
        }
    return json.dumps(response)

@form
class DBForm(CatForm):
    description = "Remote DB Connection"
    model_class = DBConnectionInfo
    start_examples = [
        "connect new database",
        "create new MySQL/PostgreSQL/SQLite connection"
    ]
    stop_examples = [
        "do not connect database",
        "stop connecting to database"
    ]
    ask_confirm = True

    def submit(self, form_data):
        conn_url = self.cat.llm(f"""You now return ONLY a data connection for a DB client to connect to a DB. Make a DB connection url from the following data: {json.dumps(form_data)}""")
        return {
            "output": f"Stringa di connessione: {conn_url}. Modifica la configurazione dalle impostazioni del DB."
        }
