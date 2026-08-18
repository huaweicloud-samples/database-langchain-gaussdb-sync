[简体中文](README.zh-CN.md) | [English](README.md)

# `database-langchain-gaussdb-sync` 用户手册

| 项目 | 适用范围 |
| --- | --- |
| 软件包 | `database-langchain-gaussdb-sync 0.1.0` |
| Python 导入包 | `langchain_gaussdb` |
| Python | 3.10～3.14 |
| LangChain | `langchain-core` |
| 数据库驱动 | 同步 `psycopg2` |
| 数据库 | 集中式和分布式 GaussDB |
| 向量索引 | 固定使用 GsDiskANN |
| 检索模式 | 集中式：Dense、BM25、Hybrid；分布式：Dense |

本项目源码位于个人仓库：
[lilee-LI/database-langchain-gaussdb-sync](https://github.com/lilee-LI/database-langchain-gaussdb-sync)。

## 目录

- [1. 手册说明](#1-手册说明)
- [2. 安装](#2-安装)
- [3. 准备数据库](#3-准备数据库)
- [4. 配置数据库连接](#4-配置数据库连接)
- [5. VectorStore 初始化与数据模型](#5-vectorstore-初始化与数据模型)
- [6. Dense 检索快速入门](#6-dense-检索快速入门)
- [7. 写入、更新、读取与删除](#7-写入更新读取与删除)
- [8. Dense、BM25 与 Hybrid 检索](#8-densebm25-与-hybrid-检索)
- [9. JSONB metadata 过滤与表达式索引](#9-jsonb-metadata-过滤与表达式索引)
- [10. ChatMessageHistory](#10-chatmessagehistory)
- [11. 异步接口兼容方式](#11-异步接口兼容方式)
- [12. 事务、并发与资源管理](#12-事务并发与资源管理)
- [13. 错误模型与安全](#13-错误模型与安全)
- [14. 测试](#14-测试)
- [15. 常见问题](#15-常见问题)
- [16. 上线检查清单](#16-上线检查清单)
- [17. 公共 API 总表](#17-公共-api-总表)
- [18. 当前版本边界](#18-当前版本边界)

## 1. 手册说明

`database-langchain-gaussdb-sync` 为 LangChain 应用提供三个核心组件：

- `GaussDBEngine`：基于 `psycopg2 ThreadedConnectionPool` 的线程安全同步执行引擎。
- `GaussDBVectorStore`：实现 LangChain `VectorStore` 合同；集中式支持 Dense、BM25、Hybrid，分布式支持 Dense，并提供 MMR、metadata 过滤和标准 Retriever。
- `GaussDBChatMessageHistory`：实现 LangChain `BaseChatMessageHistory` 合同，按 `session_id` 保存和恢复消息。

### 1.1 功能范围

| 使用需求 | 入口 |
| --- | --- |
| 创建共享数据库连接池 | `GaussDBEngine(...)` |
| 检查连接 | `engine.check_connection()` |
| 初始化向量表和当前模式所需索引 | `store.setup()` |
| 写入文本 | `store.add_texts()` |
| 写入 Document | `store.add_documents()` |
| 按 ID 读取 | `store.get_by_ids()` |
| Dense 检索 | `store.similarity_search()` |
| 带原始分数检索 | `store.similarity_search_with_score()` |
| 相关性分数检索 | `store.similarity_search_with_relevance_scores()` |
| MMR 检索 | `store.max_marginal_relevance_search()` |
| BM25 检索（集中式） | `retrieval_mode="bm25"` |
| Hybrid 检索（集中式） | `retrieval_mode="hybrid"` |
| 转为标准 Retriever | `store.as_retriever()` |
| metadata 过滤 | `filter={...}` |
| metadata 表达式索引 | `metadata_indexes={...}` |
| 删除指定文档 | `store.delete(ids=[...])` |
| 显式删除全部文档 | `store.delete(ids=None, delete_all=True)` |
| 创建聊天历史表 | `create_table=True` 或 `create_table_if_not_exists()` |
| 写入聊天消息 | `history.add_messages()` |
| 读取聊天消息 | `history.messages` |
| 清空当前会话 | `history.clear()` |
| 使用标准异步入口 | `aadd_texts()`、`asimilarity_search()`、`ainvoke()`、`aadd_messages()` |

### 1.2 设计原则

当前实现遵循以下合同：

1. 写入使用同步 `psycopg2`，不维护第二套异步 SQL 实现。
2. VectorStore 第一次非空写入会自动完成表和当前 `retrieval_mode` 所需索引的初始化。
3. `setup()` 可在部署阶段显式执行；查询路径永远不执行 DDL。
4. Dense、BM25、Hybrid 可以共用同一张表，但分别只创建本模式需要的检索索引。
5. metadata 只保存一份 JSONB，不复制为普通物理列。
6. 向量索引固定为 GsDiskANN，没有 `kind` 或自动索引类型选择。
7. 用户提供已有表时采用弱校验：适配器检查必要列名，其余交由数据库约束和 DBA 管理。
8. 集中式支持 Dense、BM25、Hybrid；分布式只支持 Dense，向量维度不得超过 1024。

## 2. 安装

### 2.1 安装发行包

发行包发布后执行：

```bash
python -m pip install database-langchain-gaussdb-sync
```

发行名称带有 `sync`，Python 导入名称仍为 `langchain_gaussdb`。

### 2.2 从个人仓库安装

```bash
git clone https://github.com/lilee-LI/database-langchain-gaussdb-sync.git
cd database-langchain-gaussdb-sync
python -m pip install -e .
```

开发和测试环境：

```bash
python -m pip install -e ".[test]"
```

运行时使用 `langchain-core`、`numpy` 和 `psycopg2`。具体解析版本以
`pyproject.toml` 为准，README 不重复维护依赖版本范围。

### 2.3 验证安装

```python
from importlib.metadata import version

from langchain_gaussdb import (
    BM25Config,
    GaussDBChatMessageHistory,
    GaussDBEngine,
    GaussDBVectorStore,
)

print("distribution:", version("database-langchain-gaussdb-sync"))
print("engine:", GaussDBEngine.__name__)
print("vectorstore:", GaussDBVectorStore.__name__)
print("history:", GaussDBChatMessageHistory.__name__)
print("bm25:", BM25Config.__name__)
```

## 3. 准备数据库

### 3.1 创建业务 schema

适配器不会创建 schema。建议由 DBA 在部署阶段创建：

```sql
CREATE SCHEMA langchain_app;
```

VectorStore 默认使用 `public`。生产环境建议显式传入 `schema_name`，避免不同连接的
`search_path` 不一致。

```python
store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
)
```

### 3.2 检查数据库能力

先确认兼容模式、事务可写状态和向量能力：

```sql
SHOW sql_compatibility;
SHOW default_transaction_read_only;
SHOW enable_vectordb;

SELECT typname
FROM pg_catalog.pg_type
WHERE lower(typname) = 'floatvector';

SELECT amname
FROM pg_catalog.pg_am
WHERE lower(amname) IN ('gsdiskann', 'bm25')
ORDER BY amname;

SELECT oprname
FROM pg_catalog.pg_operator
WHERE oprname = '###';

SELECT count(*)
FROM pg_catalog.pgxc_node
WHERE node_type = 'D';
```

至少需要：

- 连接允许执行目标业务操作。
- 数据库提供 `floatvector`。
- Dense 使用 GsDiskANN；集中式 Hybrid 同时使用 GsDiskANN 和 BM25。
- 集中式 BM25/Hybrid 使用 BM25 索引和 `###` 操作符。

部署形态与能力合同如下：

| 部署形态 | Dense | BM25 | Hybrid | `embedding_dimension` |
| --- | --- | --- | --- | --- |
| 集中式 GaussDB | 支持 | 支持 | 支持 | 1～4096 |
| 分布式 GaussDB | 支持 | 不支持 | 不支持 | 1～1024 |

适配器不会在构造函数中主动执行这一整套能力探测。初始化 BM25/Hybrid，或使用 1024 以上维度时，
适配器会查询 `pg_catalog.pgxc_node` 识别部署形态，并在任何建表/建索引 DDL 前拒绝不支持的组合。
其他数据库能力缺失时，相关 DDL 或查询会返回 `GaussDBCapabilityError` 或 `GaussDBSQLError`。

### 3.3 账号权限

VectorStore 初始化需要：

- 查看目标表列定义。
- 创建表。
- 创建 metadata 表达式索引。
- Dense/Hybrid 创建 GsDiskANN 索引。
- BM25/Hybrid 创建 BM25 索引。

运行期按实际功能需要 `SELECT`、`INSERT`、`UPDATE` 和 `DELETE`。

需要注意：`_initialized` 是每个 Python Store 实例自己的内存状态。一个新建的写实例在第一次
非空写入时仍会提交幂等 DDL。因此，当前版本的写实例需要具备初始化所需权限。只执行查询的实例
不会提交 DDL，可以复用已经由其他实例或 DBA 准备好的对象。

### 3.4 向量维度

适配器按部署形态限制向量维度：

- 集中式 GaussDB 接受 1～4096 维。
- 分布式 GaussDB 接受 1～1024 维（包含 1024）。
- 分布式配置超过 1024 时，`setup()` 或第一次非空写入在任何 DDL 前抛出
  `GaussDBCapabilityError`，不会留下半创建的表或索引。

`embedding_dimension`、文档向量长度和查询向量长度必须一致。

## 4. 配置数据库连接

### 4.1 使用环境变量保存 DSN

不要把账号凭据提交到源码仓库。PowerShell 示例：

```powershell
$env:GAUSSDB_DSN = "host=<host> port=<port> dbname=<database> user=<user> connect_timeout=10"
```

Bash 示例：

```bash
export GAUSSDB_DSN="host=<host> port=<port> dbname=<database> user=<user> connect_timeout=10"
```

密码应由 Secret Manager、CI Secret 或部署平台注入。

### 4.2 创建并检查 Engine

```python
import os

from langchain_gaussdb import GaussDBEngine

engine = GaussDBEngine(
    dsn=os.environ["GAUSSDB_DSN"],
    minconn=1,
    maxconn=10,
)

assert engine.check_connection() is True
```

`minconn` 和 `maxconn` 控制 `ThreadedConnectionPool` 的连接上下限。

### 4.3 使用连接参数 mapping

连接参数较多或密码包含特殊字符时，可使用 `connection_kwargs`：

```python
import os

from langchain_gaussdb import GaussDBEngine

engine = GaussDBEngine(
    connection_kwargs={
        "host": os.environ["GAUSSDB_HOST"],
        "port": int(os.environ.get("GAUSSDB_PORT", "8000")),
        "dbname": os.environ["GAUSSDB_DATABASE"],
        "user": os.environ["GAUSSDB_USER"],
        "password": os.environ["GAUSSDB_PASSWORD"],
        "connect_timeout": 10,
        "application_name": "database-langchain-gaussdb-sync",
    },
    minconn=1,
    maxconn=10,
)
```

`dsn` 与 `connection_kwargs` 必须二选一，不能同时提供，也不能同时省略。

### 4.4 共享 Engine

推荐让多个 Store 和 History 共享一个 Engine：

```python
dense_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
)

history = GaussDBChatMessageHistory(
    engine=engine,
    schema_name="langchain_app",
    table_name="langchain_chat_message",
    session_id="session-1",
)
```

通过 `engine=...` 注入时，Store/History 不拥有 Engine；调用它们的 `close()` 不会关闭共享池。
所有使用方结束后由创建者调用：

```python
engine.close()
```

如果直接向 Store/History 传入 `dsn` 或 `connection_kwargs`，对象会创建并拥有自己的 Engine，
此时需要调用该对象的 `close()`。

## 5. VectorStore 初始化与数据模型

### 5.1 构造函数不会执行 DDL

```python
store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
    distance_strategy="cosine",
    retrieval_mode="dense",
)
```

这一步只完成参数校验和对象构造，不创建表，不创建索引，也不访问数据库 catalog。

### 5.2 什么时候创建表和索引

存在两个初始化入口。

显式初始化：

```python
store.setup()
```

自动初始化：

```python
store.add_texts(["first non-empty write"], ids=["doc-1"])
```

第一次非空写入的顺序为：

1. 校验文本、metadata、ID 和可选 BM25 投影。
2. 调用 Embeddings 生成向量。
3. 按需识别部署形态，并校验检索模式和分布式维度限制。
4. 检查目标表；不存在时创建。
5. 检查必要列名。
6. 创建所有配置的 metadata 表达式索引。
7. `dense`/`hybrid` 创建 GsDiskANN；`bm25`/`hybrid` 创建 BM25。
8. 执行 `INSERT ... ON DUPLICATE KEY UPDATE`。

空写入直接返回空列表，不初始化：

```python
assert store.add_texts([]) == []
```

查询、`get_by_ids()` 和 `delete()` 不执行初始化。表不存在时直接查询会失败。

### 5.3 自动创建的表

默认表结构等价于：

```sql
CREATE TABLE IF NOT EXISTS langchain_app.langchain_documents (
    id text PRIMARY KEY,
    content text NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding floatvector(1024) NOT NULL
) WITH (storage_type=ustore);
```

如果 `BM25Config(column="text_lemmatized")` 指向独立文本投影，还会增加：

```sql
text_lemmatized text NULL
```

### 5.4 自动创建的索引

初始化会按 `retrieval_mode` 创建所需索引：

| 索引 | 默认字段 | 创建条件 | 用途 |
| --- | --- | --- | --- |
| 主键索引 | `id` | 新建表时 | ID 唯一性和 ODKU |
| GsDiskANN | `embedding` | Dense、Hybrid | 向量候选 |
| BM25 | `content` 或配置的文本投影 | BM25、Hybrid | 关键词候选 |
| JSONB 表达式索引 | `metadata` 中配置的 key | 配置了 `metadata_indexes` | metadata 等值、范围等过滤 |

所有适配器生成的二级索引使用确定性名称和 `CREATE INDEX IF NOT EXISTS`。

这里的“一次初始化”表示一次固定顺序，不表示所有 DDL 位于同一数据库事务中。中间步骤失败时可能已经
留下部分对象；修正原因后可以重试，幂等 DDL 会复用已经成功创建的对象。

### 5.5 retrieval_mode 决定检索索引

`dense` 只准备 GsDiskANN，`bm25` 只准备 BM25，`hybrid` 同时准备两者。这样 Dense 在分布式
GaussDB 上不会触碰不受支持的 BM25 DDL。BM25 和 Hybrid 仅支持集中式 GaussDB。

同一张表可以创建多个 Store 视图：

```python
from langchain_gaussdb import BM25Config, GaussDBVectorStore

bm25_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
    retrieval_mode="bm25",
    bm25_config=BM25Config(),
)

hybrid_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
    retrieval_mode="hybrid",
)

bm25_store.setup()
hybrid_store.setup()
```

两者共享一张表和同一份文档，不会各自保存数据；各 Store 只负责补齐本模式需要的索引。纯查询实例
不会初始化，因此它依赖目标表和本模式索引已由 DBA 或相应 Store 准备好。多个 Store 重复初始化时，
`IF NOT EXISTS` DDL 会复用已存在的同名索引。

两个实例必须保持以下配置一致：

- `schema_name` 和 `table_name`。
- 核心列名。
- `embedding_dimension`。
- `distance_strategy`。
- BM25 列定义。
- 需要使用的 `metadata_indexes`。

### 5.6 用户提供已有表的弱合同

VectorStore 对已有表只检查必要列名，不反查以下内容：

- 列类型和向量维度。
- ID 是否为主键或单列唯一键。
- 其他唯一键。
- 索引定义是否与当前配置一致。
- 表权限和拓扑能力。

这意味着已有表的正确性由用户和 DBA 负责。特别是
`ON DUPLICATE KEY UPDATE` 会由数据库根据任意命中的唯一约束触发；如果已有表存在
`UNIQUE(content)` 等竞争唯一键，相同内容的新 ID 可能更新旧行。适配器不会额外校验或改写用户约束。

## 6. Dense 检索快速入门

下面示例使用一个无外部服务依赖的演示 Embeddings。生产环境应替换为真实模型，并把
`embedding_dimension` 设置为模型输出维度。

```python
import os

from langchain_core.embeddings import Embeddings

from langchain_gaussdb import GaussDBEngine, GaussDBVectorStore


class DemoEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("gaussdb" in lowered),
            float("langchain" in lowered),
            float("database" in lowered),
        ]


engine = GaussDBEngine(
    dsn=os.environ["GAUSSDB_DSN"],
    minconn=1,
    maxconn=8,
)

store = GaussDBVectorStore(
    engine=engine,
    embedding=DemoEmbeddings(),
    embedding_dimension=3,
    schema_name="langchain_app",
    table_name="langchain_documents",
    retrieval_mode="dense",
)

try:
    store.setup()
    ids = store.add_texts(
        [
            "GaussDB provides database capabilities",
            "LangChain defines a VectorStore contract",
            "Unrelated document",
        ],
        metadatas=[
            {"topic": "gaussdb", "rank": 1},
            {"topic": "langchain", "rank": 2},
            {"topic": "other", "rank": 3},
        ],
        ids=["doc-1", "doc-2", "doc-3"],
    )
    print("ids:", ids)

    documents = store.similarity_search(
        "GaussDB with LangChain",
        k=2,
    )
    for document in documents:
        print(document.id, document.page_content, document.metadata)
finally:
    engine.close()
```

## 7. 写入、更新、读取与删除

### 7.1 add_texts

```python
ids = store.add_texts(
    ["document one", "document two"],
    metadatas=[
        {"tenant_id": "tenant-1", "rank": 1},
        {"tenant_id": "tenant-1", "rank": 2},
    ],
    ids=["doc-1", "doc-2"],
)
```

`texts`、`metadatas` 和 `ids` 长度必须一致。没有传 `ids` 时，每条文本生成一个 UUID 字符串：

```python
generated_ids = store.add_texts(["document without explicit id"])
```

调用方如果需要幂等更新，应显式提供稳定 ID。

### 7.2 add_documents

```python
from langchain_core.documents import Document

documents = [
    Document(
        id="doc-3",
        page_content="GaussDB vector retrieval",
        metadata={"topic": "vector"},
    ),
    Document(
        id="doc-4",
        page_content="GaussDB BM25 retrieval",
        metadata={"topic": "bm25"},
    ),
]

store.add_documents(documents)
```

如果 `Document.id` 为空，适配器会生成 UUID。也可以通过 `ids=[...]` 显式覆盖文档 ID。

### 7.3 ODKU 更新语义

写入 SQL 使用：

```sql
INSERT INTO <table> (<columns>)
VALUES (...)
ON DUPLICATE KEY UPDATE
    content = VALUES(content),
    metadata = VALUES(metadata),
    embedding = VALUES(embedding);
```

自动创建的表只有 `id` 主键，因此相同 ID 会更新内容、metadata、向量和可选 BM25 投影，
不会插入第二行。

适配器不在 Python 中先查 ID 是否存在，也不自己判断冲突字段；唯一性和冲突目标由 GaussDB 约束决定。

### 7.4 批次和事务边界

写入按 1000 条拆批。每个批次由 Engine 单独提交：

- 后续批次失败时，已经提交的批次不会自动回滚。
- Embeddings 返回数量必须与文本数量一致。
- 向量必须全部是有限数值，且维度正确。
- metadata 顶层必须是可序列化的字典。

需要跨多个批次的全有或全无语义时，应在业务层限制单次规模或自行设计导入事务。

### 7.5 按 ID 读取

```python
documents = store.get_by_ids(["doc-2", "doc-1", "missing"])
```

不存在的 ID 不返回占位对象。返回顺序由数据库查询结果决定，不承诺与请求顺序一致。

### 7.6 安全删除

删除指定 ID：

```python
store.delete(ids=["doc-1", "doc-2"])
```

`delete(ids=None)` 不会默认删除整表数据，而是报错。必须显式确认：

```python
store.delete(ids=None, delete_all=True)
```

`delete(ids=[])` 是幂等空操作。删除表和删除索引不属于 VectorStore 公共 API，应由 DBA 或迁移工具执行。

### 7.7 工厂方法

```python
store = GaussDBVectorStore.from_texts(
    ["one", "two"],
    embedding=embeddings,
    engine=engine,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
    ids=["doc-1", "doc-2"],
)
```

`from_texts()` 和 `from_documents()` 会立即写入，因此非空输入会触发表和当前模式所需索引的初始化。
旧的 `create_table`、`create_index` 工厂参数不再支持。

## 8. Dense、BM25 与 Hybrid 检索

本章中的 BM25 和 Hybrid 仅适用于集中式 GaussDB；分布式 GaussDB 只使用 Dense。

### 8.1 Dense 检索

```python
documents = store.similarity_search(
    "GaussDB vector index",
    k=4,
    filter={"tenant_id": {"$eq": "tenant-1"}},
)
```

也可以绕过 `embed_query()`，直接传入向量：

```python
query_vector = embeddings.embed_query("GaussDB vector index")
documents = store.similarity_search_by_vector(query_vector, k=4)
```

`distance_strategy` 支持：

| 值 | SQL 距离 | 排序 |
| --- | --- | --- |
| `cosine` | `<+>` | 越小越相似 |
| `l2` | `<->` | 越小越相似 |

### 8.2 分数语义

原始 Dense 距离：

```python
results = store.similarity_search_with_score("query", k=4)
for document, distance in results:
    print(document.id, distance)
```

Dense 原始距离越小越好。

标准相关性分数：

```python
results = store.similarity_search_with_relevance_scores("query", k=4)
for document, relevance in results:
    print(document.id, relevance)
```

相关性被归一化到 `[0, 1]`，越大越好。相关性阈值只支持 Dense：

```python
retriever = store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 8, "score_threshold": 0.75},
)
```

### 8.3 MMR

MMR 在相关性和结果多样性之间取舍：

```python
documents = store.max_marginal_relevance_search(
    "GaussDB retrieval",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,
)
```

- `fetch_k`：数据库先取回的候选数量。
- `lambda_mult=1`：更偏向相关性。
- `lambda_mult=0`：更偏向多样性。
- MMR 只支持 `retrieval_mode="dense"`。

### 8.4 BM25

```python
from langchain_gaussdb import BM25Config, GaussDBVectorStore

bm25_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
    retrieval_mode="bm25",
    bm25_config=BM25Config(),
)

keyword_docs = bm25_store.as_retriever(
    search_kwargs={"k": 4},
).invoke("GaussDB-22001")
```

默认 BM25 字段是 `content`。BM25 原始分数越大越好。

BM25 模式仍要求 `embedding` 和 Embeddings，因为表结构和写入合同与 Dense 共用；BM25 查询本身不会调用
`embed_query()`。

### 8.5 独立 BM25 文本投影

业务可以在外部完成分词、词形归一化或其他预处理：

```python
lemmatized_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="lemmatized_documents",
    retrieval_mode="bm25",
    bm25_config=BM25Config(column="text_lemmatized"),
)

lemmatized_store.add_texts(
    ["original document"],
    ids=["doc-1"],
    text_lemmatized_values=["processed document"],
)

documents = lemmatized_store.as_retriever(
    search_kwargs={
        "k": 4,
        "bm25_query": "processed query",
    }
).invoke("original query")
```

当 BM25 指向独立投影列时，每次非空写入都必须提供等长的
`text_lemmatized_values`。适配器不内置分词器。

### 8.6 Hybrid

```python
hybrid_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="langchain_documents",
    retrieval_mode="hybrid",
)

hybrid_docs = hybrid_store.as_retriever(
    search_kwargs={"k": 4},
).invoke("connection failure GaussDB-22001")
```

Hybrid 执行过程：

1. 对原始 query 生成向量并执行 Dense 检索。
2. 使用原始 query 或显式 `bm25_query` 执行 BM25 检索。
3. 两边默认各取至少 20 个候选。
4. 按文档 ID 使用加权 Reciprocal Rank Fusion 融合。
5. 返回融合分数最高的 `k` 条。

当前融合参数是内部固定配置：Dense 权重 0.7、BM25 权重 0.3、`rrf_k=60`。
Hybrid 的两次检索是两个独立数据库事务；并发写入时，两边可能看到不同快照。

### 8.7 标准 Retriever

```python
retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "filter": {"tenant_id": {"$eq": "tenant-1"}},
    },
)

documents = retriever.invoke("GaussDB")
```

本项目不增加 `as_bm25_retriever()` 或 `as_hybrid_retriever()`。先在构造函数中选择
`retrieval_mode`，再使用 LangChain 标准 `as_retriever()`。

## 9. JSONB metadata 过滤与表达式索引

### 9.1 数据模型

metadata 只存储在一个 JSONB 列中：

```python
store.add_texts(
    ["document"],
    metadatas=[
        {
            "tenant_id": "tenant-1",
            "rank": 10,
            "active": True,
            "tags": ["gaussdb", "langchain"],
            "profile": {"team": "database"},
            "optional": None,
        }
    ],
    ids=["doc-1"],
)
```

Python `None` 被保存为 JSON `null`。缺少 key 与 key 存在但值为 JSON `null` 是不同状态：

- `{"optional": {"$exists": True}}`：两者中只匹配 key 存在的文档，包括 JSON `null`。
- `{"optional": {"$exists": False}}`：只匹配缺少 key 的文档。
- `{"optional": {"$contains": None}}`：匹配 JSON `null`。

### 9.2 支持的操作符

| 类别 | 操作符 |
| --- | --- |
| 比较 | `$eq`、`$ne`、`$gt`、`$gte`、`$lt`、`$lte` |
| 集合 | `$in`、`$nin` |
| 范围 | `$between` |
| 文本 | `$like`、`$ilike` |
| JSONB 包含 | `$contains` |
| key 存在性 | `$exists` |
| 逻辑组合 | `$and`、`$or`、`$not` |

同一字典内多个字段隐式使用 AND：

```python
filter_value = {
    "tenant_id": {"$eq": "tenant-1"},
    "rank": {"$gte": 10},
}
```

显式逻辑组合：

```python
filter_value = {
    "$and": [
        {"tenant_id": {"$eq": "tenant-1"}},
        {
            "$or": [
                {"rank": {"$gte": 10}},
                {"tags": {"$contains": ["priority"]}},
            ]
        },
        {"$not": {"active": {"$eq": False}}},
    ]
}
```

Dense、BM25、Hybrid 使用同一个 filter 编译器。Hybrid 会把同一个过滤条件同时应用到 Dense 和 BM25
候选查询，避免未通过过滤的文档从另一分支泄漏到融合结果。

### 9.3 配置表达式索引

```python
indexed_store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    schema_name="langchain_app",
    table_name="indexed_documents",
    metadata_indexes={
        "tenant_id": "text",
        "rank": "bigint",
        "score": "float",
        "active": "boolean",
        "event_date": "date",
        "raw_scalar": None,
    },
)
```

支持的索引类型：

| 配置值 | 典型数据 |
| --- | --- |
| `None` | JSON 标量的精确表示 |
| `text` | 字符串 |
| `bigint` | 整数 |
| `float` | 浮点数 |
| `boolean` | 布尔值 |
| `date` | `YYYY-MM-DD` 日期 |

`integer` 会规范为 `bigint`，`double precision` 会规范为 `float`。

没有配置 `metadata_indexes` 时过滤仍然可用，但数据库可能执行顺序扫描。配置表达式索引后，过滤编译器
会生成与建索引时相同的表达式，使优化器能够选择对应索引。

所有读取同一张表的 Store 都应声明相同的 `metadata_indexes`。它不仅决定初始化 DDL，也决定查询使用的
JSONB 表达式和类型转换。

### 9.4 类型一致性

类型化表达式要求已有 metadata 数据能够转换为对应类型。例如为 `rank` 配置 `bigint` 后，
`{"rank": "not-a-number"}` 可能在建索引或查询时失败。

适配器不会扫描全部历史 JSONB 验证类型。过滤发生数据库转换错误时，会尽量转换为
`GaussDBFilterError` 并保留 SQLSTATE，但不会吞掉数据库错误。

### 9.5 JSONB null 与 SQL NULL

对类型化表达式而言，缺失 key 和 JSON `null` 通常都会在转换阶段形成 SQL NULL，因此大部分比较操作
不会匹配它们。需要区分两者时使用 `$exists`；需要匹配 JSON `null` 时使用
`$contains`。

## 10. ChatMessageHistory

### 10.1 创建与初始化

```python
from langchain_gaussdb import GaussDBChatMessageHistory

history = GaussDBChatMessageHistory(
    engine=engine,
    schema_name="langchain_app",
    table_name="langchain_chat_message",
    session_id="session-1",
    create_table=True,
)
```

也可以显式初始化：

```python
history = GaussDBChatMessageHistory(
    engine=engine,
    schema_name="langchain_app",
    table_name="langchain_chat_message",
    session_id="session-1",
)
history.create_table_if_not_exists()
```

与 VectorStore 不同，ChatMessageHistory 普通写入不会自动建表。

### 10.2 写入和读取消息

```python
from langchain_core.messages import AIMessage, HumanMessage

history.add_messages(
    [
        HumanMessage(content="GaussDB connection failed"),
        AIMessage(content="Please provide the SQLSTATE and server log."),
    ]
)

for message in history.messages:
    print(message.type, message.content)
```

也可以使用 LangChain 基类提供的便利方法：

```python
history.add_user_message("hello")
history.add_ai_message("hi")
```

### 10.3 会话隔离与清理

同一张表可以保存多个 `session_id`：

```python
another_history = GaussDBChatMessageHistory(
    engine=engine,
    schema_name="langchain_app",
    table_name="langchain_chat_message",
    session_id="session-2",
)
```

`history.messages` 只返回当前 `session_id`，按自增 `id` 升序排列。

清空当前会话：

```python
history.clear()
```

不会删除其他 session，也不会删除表。

### 10.4 表和索引

自动创建结构：

```sql
CREATE TABLE IF NOT EXISTS langchain_app.langchain_chat_message (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    message JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) WITH (storage_type=ustore);

CREATE INDEX IF NOT EXISTS <generated_name>
ON langchain_app.langchain_chat_message (session_id, id);
```

ChatMessageHistory 对自己生成的表和索引执行严格 catalog 校验。如果已存在同名但定义不同的对象，
`create_table_if_not_exists()` 会失败，而不是继续使用不兼容结构。

## 11. 异步接口兼容方式

### 11.1 底层仍是同步 psycopg2

GaussDB 数据访问使用同步 `psycopg2`。LangChain 标准异步接口通过 executor 工作线程调用对应的同步
实现，不是 driver-native async，也不是第二套 SQL 路径。

```python
await store.aadd_texts(
    ["async-compatible write"],
    ids=["async-doc-1"],
)

documents = await store.asimilarity_search(
    "async-compatible search",
    k=4,
)

documents = await hybrid_store.as_retriever(
    search_kwargs={"k": 4},
).ainvoke("hybrid query")

await history.aadd_messages(messages_to_add)
messages = await history.aget_messages()
```

### 11.2 并发配置

异步并发最终会占用 `GaussDBEngine` 的同步连接。需要根据应用 executor 并发量设置合理的
`maxconn`：

```python
engine = GaussDBEngine(
    dsn=os.environ["GAUSSDB_DSN"],
    minconn=2,
    maxconn=16,
)
```

连接池耗尽时，Engine 使用信号量等待连接，不直接把
`ThreadedConnectionPool` 的立即耗尽异常暴露给调用方。

### 11.3 取消语义

取消等待中的协程不保证取消已经在线程中运行的数据库操作。工作线程可能继续执行并提交。
需要服务端中断保证时，应配置 GaussDB 语句超时、锁超时和业务幂等策略。

当前包不提供额外的 `asetup()` 或 `aclose()`；初始化和关闭使用同步方法。

## 12. 事务、并发与资源管理

### 12.1 Engine 事务合同

`GaussDBEngine.execute()` 和 `fetch_all()` 每次使用一个池连接和一个事务：

- 成功后 commit。
- 失败后 rollback。
- 连接恢复异常时丢弃失败连接。
- SQL 和参数通过 `CompiledSQL` 分离。

需要在同一事务内执行多步逻辑时，可以使用：

```python
def operation(cursor):
    cursor.execute("SELECT 1")
    return cursor.fetchone()[0]

value = engine.transaction(
    operation,
    operation="application transaction",
)
```

不要在 `engine.transaction()` 的回调中再次调用同一个 Engine。Engine 明确拒绝同线程重入操作，
避免连接池死锁。

### 12.2 初始化和写入不是全局大事务

以下步骤分别提交：

- 建表。
- 每一个索引 DDL。
- 每一个 1000 条写入批次。

失败后重试通常可以利用 `IF NOT EXISTS` 继续，但应用不能假设失败一定“什么都没发生”。

### 12.3 Hybrid 快照

Hybrid 的 Dense 和 BM25 分支是两个独立读取事务。高并发写入下，两个候选集合可能来自不同快照。
如果业务要求严格单快照融合，需要在更高层提供隔离策略；当前公共 API 不把两个查询绑定到同一事务。

### 12.4 关闭顺序

共享 Engine：

```python
engine = GaussDBEngine(dsn=os.environ["GAUSSDB_DSN"])
store = GaussDBVectorStore(
    engine=engine,
    embedding=embeddings,
    embedding_dimension=1024,
    table_name="langchain_documents",
)

try:
    # 使用 store。
    ...
finally:
    engine.close()
```

Engine `close()` 会等待已经进入的操作结束，然后关闭连接池。不能在 Engine 自己的活动事务回调中调用
`close()`。

## 13. 错误模型与安全

### 13.1 公共错误类型

| 类型 | 说明 |
| --- | --- |
| `GaussDBError` | 统一基类 |
| `GaussDBConnectionError` | 连接源、连接池、关闭和连接恢复错误 |
| `GaussDBSQLError` | 数据库 SQL 执行错误 |
| `GaussDBSQLBuildError` | 标识符或 SQL 结构构造错误 |
| `GaussDBTransactionError` | commit、rollback 等事务错误 |
| `GaussDBCapabilityError` | 表结构或数据库能力不满足 |
| `GaussDBFilterError` | metadata filter 语法、类型或数据库转换错误 |

`GaussDBSQLError` 等数据库错误会尽可能保留 SQLSTATE，便于业务按数据库错误类别处理。

### 13.2 SQL 安全

实现遵循：

- schema、表、列和索引名使用 `psycopg2.sql.Identifier`。
- 用户值使用绑定参数，不拼接到 SQL 字符串。
- metadata key 由表达式构造器安全引用。
- DSN 和常见凭据形式在错误上下文中脱敏。

标识符仍需要满足非空、无 NUL、长度等规则。动态对象名应来自受控配置，不应直接使用未经审核的用户输入。

### 13.3 凭据和权限

- DSN 放在环境变量或 Secret Manager。
- 不在日志中打印完整连接参数。
- 按应用实际需要授权。
- 生产连接根据实例策略启用 TLS。
- E2E 测试只能连接专用可清理数据库，不能指向生产库。

## 14. 测试

### 14.1 单元测试

```bash
python -m pytest tests/unit -q
```

单元测试覆盖：

- SQL 构造和参数绑定。
- Engine 事务、连接池、并发和异常恢复。
- VectorStore 构造、初始化、写入、读取、删除。
- Dense、BM25、Hybrid、MMR 和 Retriever 合同。
- JSONB filter 运算符、类型和索引表达式。
- ChatMessageHistory 表结构和消息序列化。
- 异步入口对同步实现的委派。
- 公开导入、README、wheel 内容和类型标记。

### 14.2 只收集 E2E

没有数据库连接时，可以验证 E2E 文件能否导入和参数化：

```bash
python -m pytest tests/e2e --collect-only -q
```

`collected` 不等于真库测试通过，只能证明用例成功收集。

### 14.3 执行真库 E2E

PowerShell：

```powershell
$env:GAUSSDB_TEST_DSN = "host=<host> port=<port> dbname=<database> user=<user>"
python -m pytest -m gaussdb_e2e
```

Bash：

```bash
export GAUSSDB_TEST_DSN="host=<host> port=<port> dbname=<database> user=<user>"
python -m pytest -m gaussdb_e2e
```

E2E 场景按真实数据库部署形态执行：集中式覆盖 Dense、BM25 和 Hybrid；分布式执行 Dense，跳过
数据库明确不支持的 BM25/Hybrid 正向旅程，并验证不支持模式和 1024 以上维度会在 DDL 前被拒绝。
其余场景覆盖建表、索引、ODKU、filter、ChatMessageHistory、同步/异步适配、并发、只读策略和
隔离 wheel 安装旅程。

E2E 会创建并清理 schema、表和索引，请使用专用测试数据库。

### 14.4 代码质量

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy langchain_gaussdb
```

## 15. 常见问题

### 15.1 查询时报 relation does not exist

构造 Store 不会初始化，查询也不会执行 DDL。先用具备权限的同配置 Store 调用：

```python
store.setup()
```

或者先进行一次非空写入。

### 15.2 第一次写入报 CREATE 权限错误

第一次非空写入会创建/检查表和当前模式所需索引。当前写实例必须具有初始化所需权限。
如果对象已经由 DBA 创建，新 Store 写实例仍会提交 `CREATE INDEX IF NOT EXISTS`。

### 15.3 BM25 或 GsDiskANN 创建失败

检查：

```sql
SHOW enable_vectordb;

SELECT amname
FROM pg_catalog.pg_am
WHERE lower(amname) IN ('gsdiskann', 'bm25');
```

同时检查表是否为空、目标字段类型、索引参数、账号权限和目标拓扑限制。

### 15.4 分布式数据库使用 1024 以上维度失败

分布式 GsDiskANN 只允许 1～1024 维。适配器在初始化 DDL 前抛出 `GaussDBCapabilityError`。
把 Embeddings 模型输出和 `embedding_dimension` 一起调整到不超过 1024。

### 15.5 分布式数据库使用 BM25 或 Hybrid 失败

分布式 GaussDB 当前只支持 `retrieval_mode="dense"`。`bm25` 和 `hybrid` 会在初始化 DDL 前抛出
`GaussDBCapabilityError`；需要关键词或融合检索时使用集中式 GaussDB。

### 15.6 同一个 ID 为什么没有更新原行

自动创建表以 `id` 为主键，相同 ID 会触发 ODKU。用户表如果没有 ID 唯一约束，数据库不会把相同 ID
识别为冲突，可能插入多行。该问题属于用户提供表的约束合同。

### 15.7 表有多个唯一键时 ODKU 按哪个键

`ON DUPLICATE KEY UPDATE` 由数据库根据命中的任意唯一约束触发，不只看 ID。
用户表存在 `UNIQUE(content)` 等竞争唯一键时，可能更新由其他唯一键命中的行。
推荐 VectorStore 表只让 `id` 承担写入冲突身份。

### 15.8 metadata filter 触发类型转换错误

检查历史 JSONB 值是否都符合 `metadata_indexes` 声明的类型。类型化表达式索引不会自动清洗旧数据。

### 15.9 配置 metadata index 后仍然顺序扫描

确认：

1. 查询 Store 声明了与建表 Store 相同的 `metadata_indexes`。
2. filter 使用了与索引类型兼容的操作符和值。
3. 表中数据量足够让优化器认为索引更便宜。
4. 使用 `EXPLAIN` 检查实际表达式和执行计划。

### 15.10 MMR 在 BM25 或 Hybrid 模式报错

MMR 需要候选 embedding，只支持 `retrieval_mode="dense"`。为同一张表创建一个 Dense Store 视图后
调用 MMR。

### 15.11 直接查询 BM25 表时为什么还需要 embedding

三种模式共用同一个 VectorStore 表结构和写入合同。BM25 查询不调用 `embed_query()`，但构造函数仍需
Embedding 对象，写入仍保存向量。

### 15.12 ChatMessageHistory 报已有表定义不兼容

ChatMessageHistory 会严格检查四列顺序、类型、主键、默认值和 `(session_id, id)` 索引。
迁移或重建不兼容对象，不要绕过错误继续使用。

### 15.13 异步任务取消后数据库操作仍完成

异步 API 使用 executor 工作线程。取消协程不等于向数据库发送取消命令。使用服务端超时和幂等写入
控制影响。

### 15.14 两个不同 retrieval_mode 是否会复制数据

不会。只要 schema 和表名相同，它们读取同一张表。`retrieval_mode` 决定查询策略及初始化时需要补齐的
检索索引，不会复制文档。

### 15.15 搜索时能否临时传 retrieval_mode

不能。`retrieval_mode` 是构造参数，搜索 kwargs 中传入会被拒绝。需要另一模式时创建共享同表的
第二个 Store 视图。

## 16. 上线检查清单

### 16.1 安装与数据库

- [ ] 安装 `database-langchain-gaussdb-sync`。
- [ ] 验证公开导入和版本。
- [ ] 创建业务 schema。
- [ ] 确认连接可写状态。
- [ ] 确认 `floatvector`、GsDiskANN，以及所选模式需要的 BM25 能力。
- [ ] 确认集中式/分布式部署形态与所选 `retrieval_mode` 兼容。
- [ ] 确认向量维度符合集中式或分布式限制。

### 16.2 表和索引

- [ ] 使用目标 Embeddings 维度构造 Store。
- [ ] 由具备权限的实例执行 `setup()`。
- [ ] 确认主键、当前模式所需检索索引和 metadata 表达式索引存在。
- [ ] 对关键 filter 使用 `EXPLAIN` 验证计划。
- [ ] 用户表只保留业务真正需要的唯一约束。

### 16.3 功能

- [ ] 验证显式 ID 的重复写入符合预期。
- [ ] 验证 Dense 结果和分数方向。
- [ ] 集中式部署验证 BM25 关键词结果。
- [ ] 集中式部署验证 Hybrid 融合结果。
- [ ] 验证 JSON null、缺失 key 和 `$exists` 语义。
- [ ] 验证删除全部数据必须显式 `delete_all=True`。
- [ ] 验证不同 chat `session_id` 相互隔离。

### 16.4 并发与资源

- [ ] `maxconn` 能承载同步和异步 executor 并发。
- [ ] 共享 Engine 只由所有者关闭。
- [ ] 应用退出时等待活动数据库任务结束。
- [ ] Hybrid 两次读取允许不同快照。
- [ ] 大批量写入允许按 1000 条分批提交。

### 16.5 安全

- [ ] DSN 未写入源码、README 和日志。
- [ ] 账号按最小权限授权。
- [ ] TLS 配置与实例策略一致。
- [ ] 服务端语句超时和锁超时已配置。
- [ ] E2E 只连接专用测试数据库。

## 17. 公共 API 总表

### 17.1 根包导出

```python
from langchain_gaussdb import (
    BM25Config,
    CompiledSQL,
    GaussDBCapabilityError,
    GaussDBChatMessageHistory,
    GaussDBConnectionError,
    GaussDBEngine,
    GaussDBError,
    GaussDBFilterError,
    GaussDBSQLBuildError,
    GaussDBSQLError,
    GaussDBTransactionError,
    GaussDBVectorStore,
    __version__,
)
```

### 17.2 GaussDBEngine

| 方法 | 用途 |
| --- | --- |
| `GaussDBEngine(...)` | 创建同步连接池引擎 |
| `check_connection()` | 执行连接健康检查 |
| `connection()` | 临时借用原始同步连接 |
| `transaction(callback)` | 在单事务内执行 callback |
| `execute(compiled)` | 执行无返回行的 CompiledSQL |
| `fetch_all(compiled)` | 执行查询并返回全部行 |
| `close()` | 等待活动操作并关闭连接池 |

普通应用通常只需要 `check_connection()` 和 `close()`；SQL 执行入口主要供适配器内部使用。

### 17.3 GaussDBVectorStore

| 方法 | 用途 |
| --- | --- |
| `setup()` | 初始化表和当前模式所需索引 |
| `add_texts()` | 写入或按唯一约束更新文本 |
| `add_documents()` | 写入 Document |
| `from_texts()` | 构造并写入文本 |
| `from_documents()` | 构造并写入 Document |
| `get_by_ids()` | 按 ID 读取 |
| `delete()` | 删除指定 ID 或显式删除全部 |
| `similarity_search()` | 按实例模式检索 |
| `similarity_search_with_score()` | 返回原始距离/BM25/RRF 分数 |
| `similarity_search_by_vector()` | 按向量执行 Dense 检索 |
| `similarity_search_with_score_by_vector()` | 按向量返回 Dense 距离 |
| `similarity_search_with_relevance_scores()` | 返回 Dense 归一化相关性 |
| `max_marginal_relevance_search()` | Dense MMR |
| `max_marginal_relevance_search_by_vector()` | 按向量执行 Dense MMR |
| `as_retriever()` | LangChain 标准 Retriever |
| `close()` | 关闭对象自有 Engine |

### 17.4 GaussDBChatMessageHistory

| 方法或属性 | 用途 |
| --- | --- |
| `create_table_if_not_exists()` | 创建并严格校验表和索引 |
| `add_messages()` | 批量写入消息 |
| `add_user_message()` | 写入用户消息 |
| `add_ai_message()` | 写入 AI 消息 |
| `messages` | 读取当前 session 消息 |
| `clear()` | 清空当前 session |
| `close()` | 关闭对象自有 Engine |

## 18. 当前版本边界

- 数据库 I/O 使用同步 `psycopg2`。
- 异步 API 是 executor 兼容层，不是原生异步驱动。
- VectorStore 查询路径不执行 DDL。
- 写实例第一次非空写入会执行初始化 DDL。
- `dense` 初始化 GsDiskANN，`bm25` 初始化 BM25，`hybrid` 初始化两者。
- 集中式支持 Dense、BM25、Hybrid；分布式只支持 Dense，维度不超过 1024。
- GsDiskANN 是唯一内置向量索引类型。
- metadata 使用 JSONB 和可选表达式索引，不创建重复物理 metadata 列。
- VectorStore 对用户表采用必要列名弱校验。
- ChatMessageHistory 对自己生成的表和索引采用严格校验。
- ODKU 不在 Python 中指定冲突列，数据库任意唯一约束都可能触发更新。
- 每个写入批次独立提交，Hybrid 两个分支独立读取。
- 删除表和索引属于 DBA/迁移职责，不属于公共 API。
