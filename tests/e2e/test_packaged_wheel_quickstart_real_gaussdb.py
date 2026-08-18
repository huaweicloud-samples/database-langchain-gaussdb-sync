from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _evidence import sanitize_exception
from psycopg2 import sql

from langchain_gaussdb import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PUBLIC_SURFACE = (
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
)


@dataclass(frozen=True)
class _InstalledWheel:
    python: Path
    venv_dir: Path
    isolated_cwd: Path
    wheel: Path
    env: dict[str, str]


def _sanitized_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float = 900,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in command],
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _raise_or_note_cleanup(
    primary_error: BaseException | None,
    cleanup_errors: list[BaseException],
    *,
    scope: str,
) -> None:
    if not cleanup_errors:
        return
    if primary_error is not None:
        for error in cleanup_errors:
            sanitized = sanitize_exception(error)
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    f"{scope} cleanup also failed: "
                    f"class={sanitized['exception_class']}; "
                    f"sqlstate={sanitized['sqlstate']}; "
                    f"detail={sanitized['fragment']}"
                )
        return
    raise cleanup_errors[0]


@contextmanager
def _installed_wheel(tmp_path: Path) -> Iterator[_InstalledWheel]:
    work = tmp_path / "wheel_work"
    wheelhouse = work / "wheelhouse"
    isolated_cwd = work / "isolated_cwd"
    venv_dir = work / "venv"
    env = _sanitized_subprocess_env()
    workspace_build_artifacts = (
        PROJECT_ROOT / "build",
        PROJECT_ROOT / "dist",
        PROJECT_ROOT / "database_langchain_gaussdb_sync.egg-info",
    )
    preexisting_artifacts = {
        artifact for artifact in workspace_build_artifacts if artifact.exists()
    }
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        wheelhouse.mkdir(parents=True)
        isolated_cwd.mkdir()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                str(PROJECT_ROOT),
                "--no-deps",
                "--no-build-isolation",
                "-w",
                str(wheelhouse),
            ],
            cwd=isolated_cwd,
            env=env,
        )
        wheels = sorted(wheelhouse.glob("database_langchain_gaussdb_sync-*.whl"))
        assert len(wheels) == 1, [wheel.name for wheel in wheels]
        wheel = wheels[0]

        _run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=isolated_cwd,
            env=env,
        )
        venv_python = (
            venv_dir / "Scripts" / "python.exe"
            if os.name == "nt"
            else venv_dir / "bin" / "python"
        )
        assert venv_python.is_file()
        _run(
            [venv_python, "-m", "pip", "install", str(wheel)],
            cwd=isolated_cwd,
            env=env,
        )
        yield _InstalledWheel(
            python=venv_python,
            venv_dir=venv_dir,
            isolated_cwd=isolated_cwd,
            wheel=wheel,
            env=env,
        )
    except BaseException as exc:
        primary_error = exc
    finally:
        if work.exists():
            try:
                shutil.rmtree(work)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if work.exists():
            cleanup_errors.append(
                AssertionError(
                    "isolated wheel temporary directory remains after cleanup"
                )
            )
        for artifact in workspace_build_artifacts:
            if artifact in preexisting_artifacts or not artifact.exists():
                continue
            try:
                if artifact.is_dir():
                    shutil.rmtree(artifact)
                else:
                    artifact.unlink()
            except BaseException as exc:
                cleanup_errors.append(exc)
        _raise_or_note_cleanup(
            primary_error,
            cleanup_errors,
            scope="isolated wheel temporary environment",
        )
    if primary_error is not None:
        raise primary_error


def _payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    value = json.loads(lines[-1])
    assert isinstance(value, dict)
    return value


def _assert_installed_module(
    payload: dict[str, Any],
    installed: _InstalledWheel,
) -> None:
    module_path = Path(str(payload["module_file"])).resolve()
    venv_prefix = Path(str(payload["venv_prefix"])).resolve()
    assert module_path.is_relative_to(installed.venv_dir.resolve())
    assert venv_prefix.is_relative_to(installed.venv_dir.resolve())
    assert not module_path.is_relative_to(PROJECT_ROOT.resolve())


def _require_dense_quickstart_capability(
    control_engine: GaussDBEngine,
    capability_snapshot: dict[str, object],
) -> bool:
    if "gsdiskann" not in capability_snapshot["access_methods"]:
        pytest.skip(
            "owned wheel Dense node skipped: GaussDB capability gsdiskann unavailable"
        )
    floatvector = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = %s "
                "AND pg_type_is_visible(oid))"
            ),
            ("floatvector",),
        ),
        operation="probe parent wheel Dense floatvector capability",
    )
    if floatvector != [(True,)]:
        pytest.skip(
            "owned wheel Dense node skipped: GaussDB capability floatvector unavailable"
        )
    return (
        capability_snapshot.get("bm25") is True
        and "bm25" in capability_snapshot["access_methods"]
    )


def _cleanup_owned_db_resources(
    control_engine: GaussDBEngine,
    resource_registry: Any,
    prefix: str,
    *,
    primary_error: BaseException | None,
) -> None:
    cleanup_errors: list[BaseException] = []
    try:
        resource_registry.cleanup(primary_failed=primary_error is not None)
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        residue = control_engine.fetch_all(
            CompiledSQL(
                sql.SQL(
                    "SELECT count(*) FROM ("
                    "SELECT nspname AS name FROM pg_namespace WHERE nspname LIKE %s "
                    "UNION ALL "
                    "SELECT rel.relname FROM pg_class AS rel "
                    "JOIN pg_namespace AS ns ON ns.oid = rel.relnamespace "
                    "WHERE rel.relname LIKE %s OR ns.nspname LIKE %s) AS residue"
                ),
                (f"{prefix}%", f"{prefix}%", f"{prefix}%"),
            ),
            operation="verify parent wheel crash cleanup catalog absence",
        )
    except BaseException as exc:
        cleanup_errors.append(exc)
    else:
        if residue != [(0,)]:
            cleanup_errors.append(
                AssertionError(
                    "parent wheel cleanup left random-prefix catalog residue"
                )
            )
    _raise_or_note_cleanup(
        primary_error,
        cleanup_errors,
        scope="parent-owned wheel database resources",
    )


ROOT_IMPORT_SCRIPT = r"""
import importlib.metadata
import json
from pathlib import Path
import sys

import langchain_gaussdb as provider

expected_public_surface = [
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
version = importlib.metadata.version("database-langchain-gaussdb-sync")
module_file = Path(provider.__file__).resolve()
public_surface = list(provider.__all__)
assert public_surface == expected_public_surface
payload = {
    "module_file": str(module_file),
    "venv_prefix": sys.prefix,
    "py_typed": (module_file.parent / "py.typed").is_file(),
    "py.typed": (module_file.parent / "py.typed").is_file(),
    "version": version,
    "root_version": provider.__version__,
    "__version__": provider.__version__,
    "public_surface": public_surface,
    "__all__": public_surface,
}
print(json.dumps(payload, sort_keys=True))
"""


SYNC_QUICKSTART_SCRIPT = r"""
import json
import os
from pathlib import Path
import sys

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from psycopg2 import sql
from langchain_gaussdb import (
    CompiledSQL,
    GaussDBChatMessageHistory,
    GaussDBEngine,
    GaussDBVectorStore,
)

class QuickEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]
    async def aembed_documents(self, texts):
        return self.embed_documents(texts)
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]
    async def aembed_query(self, text):
        return self.embed_query(text)

dsn = os.environ["GAUSSDB_TEST_DSN"]
prefix = os.environ["GAUSSDB_WHEEL_PREFIX"]
schema = os.environ["GAUSSDB_WHEEL_SCHEMA"]
vector_table = os.environ["GAUSSDB_WHEEL_VECTOR_TABLE"]
chat_table = os.environ["GAUSSDB_WHEEL_CHAT_TABLE"]
engine = GaussDBEngine(dsn=dsn, minconn=1, maxconn=1)
store = None
history = None
try:
    engine.execute(
        CompiledSQL(sql.SQL("SET default_transaction_read_only=off")),
        operation="wheel quickstart writable policy",
    )
    engine.execute(
        CompiledSQL(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))),
        operation="wheel quickstart create schema",
    )
    store = GaussDBVectorStore(
        engine=engine,
        embedding=QuickEmbeddings(),
        schema_name=schema,
        table_name=vector_table,
        embedding_dimension=3,
    )
    assert store.add_texts(["wheel sync"], ids=["sync-id"]) == ["sync-id"]
    assert store.similarity_search("wheel sync", k=1)[0].id == "sync-id"
    assert store.as_retriever(search_kwargs={"k": 1}).invoke("wheel sync")[0].id == "sync-id"
    history = GaussDBChatMessageHistory(
        engine=engine,
        schema_name=schema,
        table_name=chat_table,
        session_id=f"{prefix}_session",
        create_table=True,
    )
    history.add_messages([HumanMessage(content="wheel sync chat")])
    assert [message.content for message in history.messages] == ["wheel sync chat"]
    assert store.delete(ids=["sync-id"])
    assert store.get_by_ids(["sync-id"]) == []
    store.close()
    history.close()
    assert engine.fetch_all(
        CompiledSQL(sql.SQL("SELECT %s::int"), (1,)),
        operation="wheel quickstart external Engine probe",
    ) == [(1,)]
    print(json.dumps({
        "module_file": str(Path(sys.modules["langchain_gaussdb"].__file__).resolve()),
        "venv_prefix": sys.prefix,
        "roundtrip": "sync",
    }, sort_keys=True))
finally:
    engine.close()
"""


ASYNC_QUICKSTART_SCRIPT = r"""
import asyncio
import json
import os
from pathlib import Path
import sys

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from psycopg2 import sql
from langchain_gaussdb import (
    BM25Config,
    CompiledSQL,
    GaussDBChatMessageHistory,
    GaussDBEngine,
    GaussDBVectorStore,
)

class QuickEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

async def main():
    dsn = os.environ["GAUSSDB_TEST_DSN"]
    prefix = os.environ["GAUSSDB_WHEEL_PREFIX"]
    schema = os.environ["GAUSSDB_WHEEL_SCHEMA"]
    vector_table = os.environ["GAUSSDB_WHEEL_VECTOR_TABLE"]
    chat_table = os.environ["GAUSSDB_WHEEL_CHAT_TABLE"]
    engine = GaussDBEngine(dsn=dsn, minconn=1, maxconn=1)
    try:
        engine.execute(
            CompiledSQL(sql.SQL("SET default_transaction_read_only=off")),
            operation="wheel executor async writable policy",
        )
        engine.execute(
            CompiledSQL(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))),
            operation="wheel executor async create schema",
        )
        store = GaussDBVectorStore(
            engine=engine,
            embedding=QuickEmbeddings(),
            schema_name=schema,
            table_name=vector_table,
            embedding_dimension=3,
            bm25_config=BM25Config(column="content"),
        )
        assert await store.aadd_texts(["wheel async"], ids=["async-id"]) == ["async-id"]
        assert (await store.asimilarity_search("wheel async", k=1))[0].id == "async-id"
        assert (await store.as_retriever(search_kwargs={"k": 1}).ainvoke("wheel async"))[0].id == "async-id"
        if os.environ["GAUSSDB_WHEEL_BM25_AVAILABLE"] == "1":
            bm25_store = GaussDBVectorStore(
                engine=engine,
                embedding=QuickEmbeddings(),
                schema_name=schema,
                table_name=vector_table,
                embedding_dimension=3,
                retrieval_mode="bm25",
                bm25_config=BM25Config(column="content"),
            )
            bm25_store.setup()
            bm25 = bm25_store.as_retriever(search_kwargs={"k": 1})
            assert (await bm25.ainvoke("wheel async"))[0].id == "async-id"
            bm25_branch = "executor-async-passed"
        else:
            bm25_branch = "skipped: capability bm25 unavailable"
        history = GaussDBChatMessageHistory(
            engine=engine,
            schema_name=schema,
            table_name=chat_table,
            session_id=f"{prefix}_session",
            create_table=True,
        )
        await history.aadd_messages([HumanMessage(content="wheel async chat")])
        assert [message.content for message in await history.aget_messages()] == ["wheel async chat"]
        await history.aclear()
        assert await history.aget_messages() == []
        assert await store.adelete(ids=["async-id"])
        store.close()
        history.close()
        print(json.dumps({
            "module_file": str(Path(sys.modules["langchain_gaussdb"].__file__).resolve()),
            "venv_prefix": sys.prefix,
            "roundtrip": "executor-async",
            "bm25_branch": bm25_branch,
        }, sort_keys=True))
    finally:
        engine.close()

asyncio.run(main())
"""


@pytest.mark.gaussdb_wheel_quickstart
def test_installed_wheel_root_import_public_surface(tmp_path: Path) -> None:
    with _installed_wheel(tmp_path) as installed:
        completed = _run(
            [installed.python, "-I", "-c", ROOT_IMPORT_SCRIPT],
            cwd=installed.isolated_cwd,
            env=installed.env,
        )
        payload = _payload(completed)
        _assert_installed_module(payload, installed)
        assert payload["py_typed"] is True
        assert payload["version"]
        assert payload["root_version"] == payload["version"]
        assert tuple(payload["public_surface"]) == EXPECTED_PUBLIC_SURFACE


@pytest.mark.gaussdb_wheel_quickstart
@pytest.mark.gaussdb_e2e
def test_installed_wheel_quickstart_roundtrip(
    tmp_path: Path,
    gaussdb_dsn_available: bool,
    control_engine: GaussDBEngine,
    capability_snapshot: dict[str, object],
    e2e_namespace: Any,
    resource_registry: Any,
) -> None:
    if not gaussdb_dsn_available:
        pytest.skip("GAUSSDB_TEST_DSN is not configured")
    _require_dense_quickstart_capability(control_engine, capability_snapshot)
    primary_error: BaseException | None = None
    try:
        schema = e2e_namespace.schema("wheel_sync_schema")
        vector_table = resource_registry.register_table(
            schema,
            e2e_namespace.name("wheel_sync_vector"),
        )
        chat_table = resource_registry.register_table(
            schema,
            e2e_namespace.name("wheel_sync_chat"),
        )
        with _installed_wheel(tmp_path) as installed:
            env = installed.env.copy()
            env["GAUSSDB_WHEEL_PREFIX"] = e2e_namespace.prefix
            env["GAUSSDB_WHEEL_SCHEMA"] = schema
            env["GAUSSDB_WHEEL_VECTOR_TABLE"] = vector_table.name
            env["GAUSSDB_WHEEL_CHAT_TABLE"] = chat_table.name
            completed = _run(
                [installed.python, "-I", "-c", SYNC_QUICKSTART_SCRIPT],
                cwd=installed.isolated_cwd,
                env=env,
            )
            payload = _payload(completed)
            _assert_installed_module(payload, installed)
            assert payload["roundtrip"] == "sync"
    except BaseException as exc:
        primary_error = exc
    finally:
        _cleanup_owned_db_resources(
            control_engine,
            resource_registry,
            e2e_namespace.prefix,
            primary_error=primary_error,
        )
    if primary_error is not None:
        raise primary_error


@pytest.mark.gaussdb_wheel_quickstart
@pytest.mark.gaussdb_e2e
@pytest.mark.asyncio
async def test_installed_wheel_executor_async_quickstart(
    tmp_path: Path,
    gaussdb_dsn_available: bool,
    control_engine: GaussDBEngine,
    capability_snapshot: dict[str, object],
    e2e_namespace: Any,
    resource_registry: Any,
) -> None:
    if not gaussdb_dsn_available:
        pytest.skip("GAUSSDB_TEST_DSN is not configured")
    bm25_available = _require_dense_quickstart_capability(
        control_engine,
        capability_snapshot,
    )
    primary_error: BaseException | None = None
    try:
        schema = e2e_namespace.schema("wheel_async_schema")
        vector_table = resource_registry.register_table(
            schema,
            e2e_namespace.name("wheel_async_vector"),
        )
        chat_table = resource_registry.register_table(
            schema,
            e2e_namespace.name("wheel_async_chat"),
        )
        with _installed_wheel(tmp_path) as installed:
            env = installed.env.copy()
            env["GAUSSDB_WHEEL_PREFIX"] = e2e_namespace.prefix
            env["GAUSSDB_WHEEL_SCHEMA"] = schema
            env["GAUSSDB_WHEEL_VECTOR_TABLE"] = vector_table.name
            env["GAUSSDB_WHEEL_CHAT_TABLE"] = chat_table.name
            env["GAUSSDB_WHEEL_BM25_AVAILABLE"] = "1" if bm25_available else "0"
            completed = _run(
                [installed.python, "-I", "-c", ASYNC_QUICKSTART_SCRIPT],
                cwd=installed.isolated_cwd,
                env=env,
            )
            payload = _payload(completed)
            _assert_installed_module(payload, installed)
            assert payload["roundtrip"] == "executor-async"
            expected_bm25 = (
                "executor-async-passed"
                if bm25_available
                else "skipped: capability bm25 unavailable"
            )
            assert payload["bm25_branch"] == expected_bm25
    except BaseException as exc:
        primary_error = exc
    finally:
        _cleanup_owned_db_resources(
            control_engine,
            resource_registry,
            e2e_namespace.prefix,
            primary_error=primary_error,
        )
    if primary_error is not None:
        raise primary_error
