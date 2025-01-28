from cat.mad_hatter.decorators import tool, hook, plugin
from cat.log import log
from cat.experimental.form import CatForm, CatFormState, form

from typing import List
from pydantic import BaseModel, ValidationError

import json
import os
import sqlparse

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cat.plugins.purrsql.models import HelperLLM, PurrSQLSettings, DBConnectionInfo, EXAMPLE_DB_URL
from cat.plugins.purrsql.helpers import clean_langchain_query, extract_columns_from_query

@hook
def agent_prompt_prefix(prefix, cat):
    return """You are a DB client. You reply in a complete and precise way to user questions.
You can query a SQLite database from an ordering system.
When suited, provide the data in a markdown table format, with the first row being the key names, else provide the data in a human readable format.
"""

db = None
custom_llm = None
enable_query_debugger = True

def apply_settings(settings):
    global db, custom_llm, enable_query_debugger

    try:
        db = SQLDatabase.from_uri(settings["db_url"])
    except Exception as e:
        log.error(f"Failed to connect to the database: {e}")
        db = None

    match settings["helper_llm"]:
        case HelperLLM.llama:
            #TODO
            pass
        case HelperLLM.gemini:
            os.environ["GOOGLE_API_KEY"] = settings["helper_llm_api_key"]
            from langchain_google_genai import ChatGoogleGenerativeAI
            custom_llm = ChatGoogleGenerativeAI(
                model=settings["helper_llm_model"],
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
        case _:
            custom_llm = None
    
    enable_query_debugger = settings["enable_query_debugger"]

@plugin
def settings_model():
    return PurrSQLSettings

@plugin
def save_settings(settings):
    apply_settings(settings)

@hook  # default priority = 1
def after_cat_bootstrap(cat):
    settings = cat.mad_hatter.get_plugin().load_settings()

    # Get the default settings from the settings model schema
    default_settings = PurrSQLSettings.model_json_schema()["properties"]
    default_settings = {
        k: v.get("default", "") for k, v in default_settings.items() 
        if not k.startswith("_")
    }

    # Check if the settings are missing or incomplete (settings file updated manually or old plugin version)
    if not settings or not all(key in settings for key in default_settings):
        # Merge existing settings with defaults, keeping existing values when present
        settings = {**default_settings, **(settings or {})}
        cat.mad_hatter.get_plugin().save_settings(settings)
    
    apply_settings(settings)


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
        "start new database connection",
        "start new MySQL/PostgreSQL/SQLite connection"
    ]
    stop_examples = [
        "do not connect database",
        "stop connecting to database"
    ]
    ask_confirm = True
    
    # Implement custom form messages
    def message(self): # 
        if self._state == CatFormState.CLOSED: #
            return {
                "output": "Connection wizard completed. Bye!"
            }
        
        separator = "\n - "
        missing_fields = ""
        if self._missing_fields:
            missing_fields = "\nSome fields are missing:"
            missing_fields += separator + separator.join(self._missing_fields)
        invalid_fields = ""
        if self._errors:
            invalid_fields = "\nSome information you provided is invalid:"
            invalid_fields += separator + separator.join(self._errors)

        info_list = ""
        if self._model:
            info_list = separator + separator.join([f"{k}: {v}" for k, v in self._model.items()])
            out = f"""Connection settings provided until now:{info_list}<br>{missing_fields}{invalid_fields}"""
        
        if self._state == CatFormState.WAIT_CONFIRM:
            out += "\n --> Write 'yes' to confirm and connect, 'no' to cancel."

        return {
            "output": out
        }
    
    # Using custom validator, since optional fields are always ignored in current implementation
    def validate(self):
        self._missing_fields = []
        self._errors = []

        try:
            # First validate basic fields
            if "db_type" in self._model:
                # Check if type is valid, if not use LLM to suggest correction
                allowed_types = ["mysql", "postgresql", "sqlite"]

                if self._model["db_type"].lower() not in allowed_types:
                    # Useful when user types "sqlite3" instead of "sqlite"
                    # Works if I ask to connect to "that DB with a delfin in the logo and name that start with M". Useless but fun.
                    suggestion = self.cat.llm(f"Which database type from {allowed_types} is most similar to '{self._model['db_type']}'? Reply with just the name in lowercase or 'invalid' if none match.")
                    if suggestion.strip().lower() in allowed_types:
                        self._model["db_type"] = suggestion.strip().lower()
                
                # Remove database authentication fields for SQLite since they are not needed
                if self._model["db_type"].lower() == "sqlite":
                    # Using dictionary comprehension to filter out fields
                    self._model = {k: v for k, v in self._model.items() 
                                 if k not in ["db_user", "db_password", "db_name"]}
            
            if "db_port" in self._model:
                # Check if port is valid, if not use LLM to suggest correction
                if self._model["db_port"] < 0 or self._model["db_port"] > 65535:
                    self._errors.append("Port number must be between 0 and 65535")
                    del self._model["db_port"]
            
            # Attempts to create the model object to update the default values and validate it
            self.model_getter()(**self._model).model_dump(mode="json")

            # If model is valid change state to COMPLETE
            self._state = CatFormState.COMPLETE

        except ValidationError as e:
            # Collect ask_for and errors messages
            for error_message in e.errors():
                field_name = error_message["loc"][0]
                if error_message["type"] == "missing":
                    self._missing_fields.append(field_name)
                else:
                    self._errors.append(f'{field_name}: {error_message["msg"]}')
                    del self._model[field_name]

            # Set state to INCOMPLETE
            self._state = CatFormState.INCOMPLETE

    # This method is called when all fields are filled and the form is confirmed
    def submit(self, form_data):
        conn_url = clean_langchain_query(
            self.cat.llm(f"""You now return ONLY a data connection for a DB client to connect to a DB. Make a DB connection url from the following data: {json.dumps(form_data)}""")
        )
        
        settings_updated = False

        try:
            settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
            settings = {}
            with open(settings_path, "r") as f:
                settings = json.load(f)
                settings["db_url"] = conn_url
            with open(settings_path, "w") as f:
                json.dump(settings, f, indent=4)
            
            apply_settings(settings)
            settings_updated = True
        except Exception as e:
            log.error(f"Failed to update the settings from runtime: {e}")
        
        if settings_updated:
            if enable_query_debugger:
                self.cat.send_chat_message(f"""Stringa di connessione: \n```{conn_url}```""")
            return {
                "output": "La configurazione è stata aggiornata."
            }
        else:
            return {
                "output": f"Errore durante l'aggiornamento della configurazione. Impostare manualmente la stringa di connessione dalle impostazioni:\n```{conn_url}```"""
            }
