from importlib import metadata

from langchain_gaussdb.chat_message_history import GaussDBChatMessageHistory
from langchain_gaussdb.engine import GaussDBEngine
from langchain_gaussdb.errors import (
    GaussDBCapabilityError,
    GaussDBConnectionError,
    GaussDBError,
    GaussDBFilterError,
    GaussDBSQLBuildError,
    GaussDBSQLError,
    GaussDBTransactionError,
)
from langchain_gaussdb.hybrid_search import BM25Config
from langchain_gaussdb.sql import CompiledSQL
from langchain_gaussdb.vectorstore import GaussDBVectorStore

try:
    __version__ = metadata.version("database-langchain-gaussdb-sync")
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "__version__",
    "BM25Config",
    "CompiledSQL",
    "GaussDBEngine",
    "GaussDBVectorStore",
    "GaussDBChatMessageHistory",
    "GaussDBError",
    "GaussDBConnectionError",
    "GaussDBSQLError",
    "GaussDBSQLBuildError",
    "GaussDBTransactionError",
    "GaussDBCapabilityError",
    "GaussDBFilterError",
]
