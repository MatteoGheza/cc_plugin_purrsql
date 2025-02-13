from cat.mad_hatter.decorators import tool, hook, plugin
from cat.log import log
from cat.experimental.form import CatForm, CatFormState, form

from pydantic import ValidationError

import json
import os
import sqlparse

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cat.plugins.purrsql.models import HelperLLM, PurrSQLSettings, DBConnectionInfo
from cat.plugins.purrsql.helpers import clean_langchain_query, extract_columns_from_query

@hook
def agent_prompt_prefix(prefix, cat):
    global tables_list
    db_tables = ", ".join(tables_list) if tables_list else "unknown tables"
    return f"""You are a DB client. You reply in a complete and precise way to user questions.
You can query a database and retrieve data from it.
When suited, provide the data in a markdown table format, with the first row being the key names, else provide the data in a human readable format.
The DB tables you can query are: {db_tables}.
"""

db = None
tables_list = []
custom_llm = None
enable_query_debugger = True

def save_db_table_names():
    global db, tables_list
    if db is not None:
        tables_list = db.get_usable_table_names()

def apply_settings(settings):
    global db, custom_llm, enable_query_debugger

    try:
        db_url = settings["db_url"]
        if db_url.startswith("mysql://"):
            db_url = db_url.replace("mysql://", "mysql+pymysql://")
        db = SQLDatabase.from_uri(db_url)
        save_db_table_names()
    except Exception as e:
        log.error(f"Failed to connect to the database: {e}")
        db = None

    match settings["helper_llm"]:
        case HelperLLM.llama:
            from langchain_ollama import ChatOllama
            custom_llm = ChatOllama(
                model=settings["helper_llm_model"],
                base_url=settings["helper_llm_base_url"],
                temperature=0.7,
                max_retries=2
            )
            pass
        case HelperLLM.gemini:
            os.environ["GOOGLE_API_KEY"] = settings["helper_llm_api_key"]
            from langchain_google_genai import ChatGoogleGenerativeAI
            custom_llm = ChatGoogleGenerativeAI(
                model=settings["helper_llm_model"],
                temperature=0.7,
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
    # Hacky workaround: there is no way to get a callback when the settings are saved from the api,
    # so we override the save_settings method to save the settings to a file.
    # This is not recommended for production use, as it will not work in a multi-instance environment.
    # The only alternative is to check settings for changes at every prompt_prefix call, but that is not efficient.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    settings_path = os.path.join(current_dir, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
    return settings

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
def improve_natural_language_queries(tool_input, cat):
    """Use this plugin whenever a user requests data of one of the DB tables, if not explicitly asking to query the DB.
Tool input is the user request, in natural language, without any updates.
Examples:
- Get all albums
- What are the most frequent nationality states of the customers?
"""
    return f"Query the db to {tool_input}"


@tool(
    examples=[
        "Show me the list of products",
        "Query orders ordered by added date",
        "Show me the list of products with price less than 10",
        "Get all the products."
    ]
)
def query_database_data(tool_input, cat):
    """Use this to perform database operations such as retrieving, inserting, updating, filtering, or deleting data.
The plugin interprets natural language commands (not SQL) and generates the appropriate JSON response.
tool_input is a HUMAN FORMATTED STRING, WHICH IS A QUESTION OR COMMAND, NOT AN SQL QUERY OR ANYTHING ELSE.
The command can be multiple requests, separated by "THEN", which will be executed in order.
If the query returns error, the "result" key contains the error message string.
PROVIDE THE DATA IN A MARKDOWN TABLE FORMAT, WITH THE FIRST ROW BEING THE KEY NAMES.
Example response:
{"result": "[('Acqua Naturale 0.5L', 1), ('Birra Bionda', 3.5)", "columns": ["name", "price"]}
"""
    global db, custom_llm, enable_query_debugger

    if db is None:
        return "Database is not connected. Please update the settings or ask 'check the DB connection' to do troubleshooting."
    
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
- If no limit is specified, add a limit of 50 rows

2. If there are any mistakes in a query, rewrite that specific query.
3. If there are no mistakes in a query, reproduce it exactly as is.

Important rules:
- Process and output ALL queries in the original order
- Maintain the original sequence of operations
- Each query must be separated by a semicolon and combined into a single string
- Include a semicolon after the last query
- Do not include any comments in the output
- Do not skip or ignore any queries
- When asking for a generic resource or counting groups, do not select only the id but also, if possible, the name
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
                "columns": extract_columns_from_query(cat._llm, query)
            }
        else:
            response = []
            for statement in statements:
                # Ignore BEGIN and COMMIT statements if added by Langchain
                if not statement.startswith("BEGIN") and not statement.startswith("COMMIT"):
                    response.append({
                        "result": str(db.run(statement)),
                        "columns": extract_columns_from_query(cat._llm, statement)
                    })
    except Exception as e:
        response = {
            "result": str(e)
        }
    return json.dumps(response)


@tool
def check_db_connection(tool_input, cat):
    """This plugin should be used when user asks if database connection works."""
    global db, tables_list
    settings = cat.mad_hatter.get_plugin().load_settings()

    if db is None:
        try:
            SQLDatabase.from_uri(settings["db_url"])
        except Exception as e:
            return f"Database connection failed: {e}"
    else:
        # Check if the connection is still alive
        try:
            db.run("SELECT 1")
        except Exception as e:
            return f"Database connection failed: {e}"
        
        save_db_table_names()
        
        return json.dumps({
            "tables": tables_list
        })


@form
class DBConnectionURIForm(CatForm):
    description = "Generate a new database connection URI that can be used in settings to connect to a database."
    model_class = DBConnectionInfo
    start_examples = [
        "generate new database connection URI",
        "generate new MySQL/PostgreSQL/SQLite connection URL"
    ]
    stop_examples = [
        "stop generating connection URL",
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
        else:
            return {
                "output": "Please provide the database connection information to generate connection URI."
            }
        
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
        return {
            "output": f"""Connection URL: \n```{conn_url}```"""
        }
