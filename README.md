[简体中文](README.zh-CN.md) | English

# database-langchain-gaussdb-sync

`database-langchain-gaussdb-sync` provides production-oriented GaussDB
integrations for LangChain. The distribution name intentionally includes
`sync` because database access uses the synchronous psycopg2 driver. The
Python import package remains `langchain_gaussdb`:

```python
from langchain_gaussdb import (
    BM25Config,
    GaussDBChatMessageHistory,
    GaussDBEngine,
    GaussDBVectorStore,
)
```

The package includes a synchronous psycopg2 connection-pool engine, a LangChain
`VectorStore`, and a `BaseChatMessageHistory` implementation. Centralized
GaussDB supports dense, BM25, and hybrid retrieval; distributed GaussDB supports
dense retrieval only. LangChain's standard asynchronous APIs remain available
through executor workers over the synchronous implementation.

| Deployment | Dense | BM25 | Hybrid | Vector dimensions |
| --- | --- | --- | --- | --- |
| Centralized GaussDB | Yes | Yes | Yes | 1-4096 |
| Distributed GaussDB | Yes | No | No | 1-1024 |

## Install

After publishing the distribution:

```bash
python -m pip install database-langchain-gaussdb-sync
```

For local development and tests:

```bash
git clone https://github.com/lilee-LI/database-langchain-gaussdb-sync.git
cd database-langchain-gaussdb-sync
python -m pip install -e ".[test]"
```

The runtime dependencies are `langchain-core`, `numpy>=1.26,<3.0`, and
`psycopg2`. Python 3.10 through 3.14 is supported.

## Connection

Keep credentials in an environment variable or secret manager. The values
below are placeholders:

```powershell
$env:GAUSSDB_DSN = "host=<host> port=<port> dbname=<database> user=<user>"
```

```python
import os

from langchain_gaussdb import GaussDBEngine

engine = GaussDBEngine(
    dsn=os.environ["GAUSSDB_DSN"],
    minconn=1,
    maxconn=10,
)
```

`GaussDBEngine` owns a psycopg2 `ThreadedConnectionPool`. A single engine can
be shared by multiple vector stores and chat histories. Close the engine only
after all objects using it have finished.

## VectorStore

Provide any LangChain `Embeddings` implementation whose vector size matches
`embedding_dimension`:

```python
from langchain_gaussdb import GaussDBVectorStore

store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="public",
    table_name="langchain_documents",
)

ids = store.add_texts(
    ["GaussDB works with LangChain"],
    metadatas=[{"topic": "intro", "tenant_id": "tenant-1"}],
    ids=["doc-1"],
)
documents = store.similarity_search("LangChain on GaussDB", k=4)
```

The normal write path uses automatic initialization. The first non-empty
write creates a missing table and prepares the indexes required by the store's
`retrieval_mode` in one pass. Searches never execute table or index DDL.

For deployment-time initialization, call `setup()` explicitly while using an
account with DDL permission:

```python
store.setup()
```

One initialization performs this fixed sequence:

1. Create the table when it is missing, then check only that the required
   column names exist.
2. Create every configured JSONB expression index.
3. Create the GsDiskANN vector index for `dense` or `hybrid`.
4. Create the BM25 index for `bm25` or `hybrid`.

Index creation is idempotent and uses `CREATE INDEX IF NOT EXISTS`. Existing
user-managed tables follow a weak contract: the adapter checks required
column names, while GaussDB remains responsible for types, constraints,
permissions, index conflicts, and unsupported DDL.

The generated table uses a text primary key, text content, JSONB metadata,
and `floatvector(embedding_dimension)`. Writes use
`INSERT ... ON DUPLICATE KEY UPDATE`, so an existing primary-key value is
updated by the database.

The vector index is always GsDiskANN and follows `distance_strategy`, which
may be `"cosine"` or `"l2"`. The adapter accepts dimensions from 1 through
4096 on centralized GaussDB and from 1 through 1024 on distributed GaussDB.
The distributed limit is enforced before any table or index DDL.

## Metadata filters and indexes

Metadata is stored once in JSONB. Use the standard `filter={...}` search
argument:

```python
documents = store.similarity_search(
    "index initialization",
    k=4,
    filter={
        "$and": [
            {"tenant_id": {"$eq": "tenant-1"}},
            {"rank": {"$gte": 10}},
        ]
    },
)
```

Supported operators are:

- Comparison: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`
- Membership and range: `$in`, `$nin`, `$between`
- Text and JSON: `$like`, `$ilike`, `$contains`
- Presence: `$exists`
- Logical composition: `$and`, `$or`, `$not`

Optional `metadata_indexes` create JSONB expression indexes without adding
duplicate physical columns:

```python
indexed_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    table_name="indexed_documents",
    metadata_indexes={
        "tenant_id": "text",
        "event_date": "date",
        "rank": "bigint",
    },
)
```

Use `None` for representation-exact JSON scalar equality, or use `text`,
`bigint`, `float`, `boolean`, `date`, `time`, or `timestamp` for typed equality
and range filtering. `$exists` checks key presence, so a key containing JSON
`null` still exists. `$contains` uses the JSONB containment operator.

## BM25 And Hybrid Retrieval

`retrieval_mode` selects both the query algorithm and the retrieval indexes
prepared during initialization. `dense` creates GsDiskANN, `bm25` creates BM25,
and `hybrid` creates both. BM25 and hybrid are rejected on distributed GaussDB
before initialization DDL. Use LangChain's standard `as_retriever()` entry
point:

```python
from langchain_gaussdb import BM25Config, GaussDBVectorStore

bm25_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    table_name="langchain_documents",
    retrieval_mode="bm25",
    bm25_config=BM25Config(),
)
hybrid_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    table_name="langchain_documents",
    retrieval_mode="hybrid",
)

keyword_docs = bm25_store.as_retriever(search_kwargs={"k": 4}).invoke(
    "GaussDB-22001"
)
hybrid_docs = hybrid_store.as_retriever(search_kwargs={"k": 4}).invoke(
    "connection failure GaussDB-22001"
)
```

By default, BM25 is built from the content column. For an externally
preprocessed text column, configure the column and supply its stored and query
projections:

```python
lemmatized_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    table_name="lemmatized_documents",
    retrieval_mode="bm25",
    bm25_config=BM25Config(column="text_lemmatized"),
)
lemmatized_store.add_texts(
    ["original text"],
    text_lemmatized_values=["processed text"],
)
documents = lemmatized_store.as_retriever(
    search_kwargs={"bm25_query": "processed query"}
).invoke("original query")
```

Dense distance scores are lower-is-better. Dense relevance scores are
normalized to `[0, 1]`, where higher is better. BM25 scores are higher-is-better.
Hybrid reciprocal-rank-fusion scores are also higher-is-better.

## ChatMessageHistory

```python
from langchain_gaussdb import GaussDBChatMessageHistory

history = GaussDBChatMessageHistory(
    engine=engine,
    schema_name="public",
    table_name="langchain_chat_messages",
    session_id="session-1",
    create_table=True,
)
history.add_user_message("hello")
history.add_ai_message("hi")
messages = history.messages
```

Unlike `GaussDBVectorStore`, chat-history table creation is explicitly
controlled by `create_table=True` or `create_table_if_not_exists()`.

## Standard Async Compatibility

GaussDB access in this package uses psycopg2. Its standard database API is
synchronous, so this package does not provide driver-native asynchronous
database I/O. LangChain's standard async APIs delegate the corresponding
synchronous embedding and database implementation to an executor worker:

```python
await store.aadd_texts(["async-compatible write"])
documents = await store.asimilarity_search("async-compatible search")
documents = await hybrid_store.as_retriever().ainvoke("hybrid query")
await history.aadd_messages(messages_to_add)
```

Each executor worker uses the same psycopg2 `ThreadedConnectionPool`. The async
methods are compatibility wrappers over the synchronous implementation, not a
second async SQL path. Cancellation does not guarantee server-side query
cancellation; work already running may finish in its worker and commit.
Configure database timeouts when server-side interruption is
required. `close()` is synchronous; the package does not add `asetup()` or
`aclose()` methods.

## Delete safety

`delete(ids=None)` does not delete every row by default:

```python
store.delete(ids=["doc-1"])
store.delete(ids=None, delete_all=True)
```

This intentionally differs from the LangChain VectorStore contract, where an
implementation may interpret `ids=None` as deleting all rows. Dropping tables
and administering indexes belong in migrations or DBA tooling.

## Tests

Run the functional unit suite:

```bash
python -m pytest tests/unit -q
```

Collect the real-database suite without connecting:

```bash
python -m pytest tests/e2e --collect-only -q
```

To execute real GaussDB tests, provide a dedicated disposable database:

```powershell
$env:GAUSSDB_TEST_DSN = "host=<host> port=<port> dbname=<database> user=<user>"
python -m pytest -m gaussdb_e2e
```

The E2E suite creates isolated schemas, tables, and indexes and cleans them up
after each run. Do not point it at a production database.

## Initial-version boundaries

- Database I/O uses synchronous psycopg2 connections.
- Vector index creation is fixed to GsDiskANN; there is no index-kind switch
  or topology fallback.
- Centralized GaussDB supports `dense`, `bm25`, and `hybrid`; distributed
  GaussDB supports `dense` only and limits `embedding_dimension` to 1024.
- Query methods never perform DDL. Prepare objects with the first write or an
  explicit deployment-time initialization.
- Existing table validation checks required names only; user-managed schema
  compatibility remains the user's responsibility.
- BM25 and hybrid retrieval use the configured text projection and the same
  JSONB filter compiler as dense retrieval.

## Security

Identifiers are quoted, values are passed as bound parameters, and connection
errors redact common credential forms. Keep DSNs in environment variables or
a secret manager, use TLS according to your GaussDB deployment policy, and
grant application accounts only the permissions they require.
