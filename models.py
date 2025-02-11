from pydantic import BaseModel
from enum import Enum

EXAMPLE_DB_URL = "sqlite:///cat/plugins/purrsql/example.db"

class HelperLLM(str, Enum):
    cat = "cat"
    llama = "llama"
    gemini = "gemini"

class PurrSQLSettings(BaseModel):
    db_url: str = EXAMPLE_DB_URL
    enable_query_debugger: bool = True
    helper_llm_api_key: str = ""
    helper_llm_model: str = ""
    helper_llm_base_url: str = ""
    helper_llm: HelperLLM = HelperLLM.cat

class DBType(str, Enum):
    sqlite = "sqlite"
    mysql = "mysql"
    postgresql = "postgresql"

class DBConnectionInfo(BaseModel):
    db_type: DBType = DBType.sqlite
    db_host_or_path: str
    db_port: int = 0
    db_name: str = ""
    username: str = ""
    password: str = ""
