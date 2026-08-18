from __future__ import annotations

import asyncio
import hashlib
import os
import queue
import re
import threading
import uuid
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime

import pytest
from _evidence import CleanupEvidence, sanitize_exception
from langchain_core.embeddings import Embeddings
from psycopg2 import extensions, sql

from langchain_gaussdb import GaussDBEngine
from langchain_gaussdb.sql import CompiledSQL

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[object]):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)


@pytest.fixture(scope="session")
def gaussdb_dsn_available() -> bool:
    return bool(os.environ.get("GAUSSDB_TEST_DSN"))


@pytest.fixture(scope="session", autouse=True)
def configure_gaussdb_index_build_memory(
    gaussdb_dsn_available: bool,
) -> Iterator[None]:
    """Apply deterministic encoding and GsDiskANN memory to E2E connections."""
    if not gaussdb_dsn_available:
        yield
        return

    original_dsn = os.environ["GAUSSDB_TEST_DSN"]
    configured_options = extensions.parse_dsn(original_dsn).get("options", "")
    os.environ["GAUSSDB_TEST_DSN"] = extensions.make_dsn(
        original_dsn,
        options=(
            f"{configured_options} -c client_encoding=UTF8 "
            "-c maintenance_work_mem=128MB"
        ).strip(),
    )
    try:
        yield
    finally:
        os.environ["GAUSSDB_TEST_DSN"] = original_dsn


def _verify_session_setting(
    engine: GaussDBEngine,
    *,
    verification_query: str,
    expected: list[tuple[object, ...]],
) -> None:
    observed = engine.fetch_all(CompiledSQL(sql.SQL(verification_query)))
    if observed != expected:
        raise AssertionError("test Engine session policy verification failed")


def _close_engine_bounded(
    engine: GaussDBEngine,
    *,
    timeout: float = 10,
) -> None:
    failures: queue.Queue[BaseException] = queue.Queue()

    def close() -> None:
        try:
            engine.close()
        except BaseException as exc:
            failures.put(exc)

    worker = threading.Thread(
        target=close,
        name="gaussdb-e2e-engine-close",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError("test Engine close exceeded bounded teardown timeout")
    if not failures.empty():
        raise failures.get_nowait()


@pytest.fixture
def gaussdb_engine_factory(
    request: pytest.FixtureRequest,
    gaussdb_dsn_available: bool,
    e2e_prefix: str,
) -> Iterator[Callable[..., GaussDBEngine]]:
    if not gaussdb_dsn_available:
        pytest.skip("GAUSSDB_TEST_DSN is not configured")

    engines: list[GaussDBEngine] = []

    def create(
        *,
        minconn: int = 1,
        maxconn: int = 4,
        read_only: bool = True,
    ) -> GaussDBEngine:
        base_dsn = os.environ["GAUSSDB_TEST_DSN"]
        configured_options = extensions.parse_dsn(base_dsn).get("options", "")
        read_only_option = (
            f"-c default_transaction_read_only={'on' if read_only else 'off'}"
        )
        # ThreadedConnectionPool may close connections above minconn when they
        # are returned.  Startup options, unlike a one-time SET on existing
        # sessions, also apply to replacement connections created later.
        dsn = extensions.make_dsn(
            base_dsn,
            application_name=e2e_prefix,
            options=(f"{configured_options} {read_only_option}").strip(),
        )
        engine = GaussDBEngine(dsn=dsn, minconn=minconn, maxconn=maxconn)
        engines.append(engine)
        return engine

    yield create

    failures: list[BaseException] = []
    for engine in reversed(engines):
        try:
            _close_engine_bounded(engine)
        except BaseException as exc:
            failures.append(exc)
    if failures:
        report = getattr(request.node, "rep_call", None)
        message = f"{len(failures)} test Engine instance(s) failed sanitized cleanup"
        if report is not None and report.failed:
            warnings.warn(message, RuntimeWarning, stacklevel=1)
        else:
            raise AssertionError(message) from None


@pytest.fixture
def writable_engine_factory(
    gaussdb_engine_factory: Callable[..., GaussDBEngine],
) -> Callable[..., GaussDBEngine]:
    def create(*, minconn: int = 1, maxconn: int = 4) -> GaussDBEngine:
        return gaussdb_engine_factory(
            minconn=minconn,
            maxconn=maxconn,
            read_only=False,
        )

    return create


@pytest.fixture
def configure_gsdiskann_session_memory() -> Callable[[GaussDBEngine], None]:
    def configure(engine: GaussDBEngine) -> None:
        _verify_session_setting(
            engine,
            verification_query="SHOW maintenance_work_mem",
            expected=[("128MB",)],
        )

    return configure


@pytest.fixture
def control_engine(
    writable_engine_factory: Callable[..., GaussDBEngine],
) -> GaussDBEngine:
    return writable_engine_factory(minconn=1, maxconn=1)


@dataclass(frozen=True)
class E2ENamespace:
    prefix: str
    registry: ResourceRegistry

    def name(self, role: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", role):
            raise ValueError("resource role must be a lowercase identifier")
        suffix = f"_{role}"
        return f"{self.prefix[: 63 - len(suffix)]}{suffix}"

    def schema(self, role: str = "schema") -> str:
        return self.registry.register_schema(self.name(role))

    def table(self, schema: str, role: str) -> TableResource:
        return self.registry.register_table(schema, self.name(role))

    def index(self, schema: str, role: str) -> str:
        return self.registry.register_index(schema, self.name(role))

    def session_id(self, role: str = "session") -> str:
        return self.registry.register_session_id(self.name(role))


@dataclass(frozen=True)
class TableResource:
    schema: str
    name: str


@dataclass(frozen=True)
class _Resource:
    kind: str
    name: str
    schema: str | None = None
    backend_pid: int | None = None
    backend_start: datetime | None = None
    application_name: str | None = None


class ResourceRegistry:
    """Own only objects created with the current test's random prefix."""

    def __init__(
        self,
        prefix: str,
        control: GaussDBEngine,
        *,
        owner_test: str,
    ) -> None:
        self.prefix = prefix
        self.owner_test = owner_test
        self._control = control
        self._resources: list[_Resource] = []
        self.evidence: list[CleanupEvidence] = []
        self._lock = threading.RLock()

    def _validate(self, name: str) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(name) or not name.startswith(self.prefix):
            raise ValueError("refusing to register a resource outside this E2E prefix")

    def register_schema(self, name: str) -> str:
        self._validate(name)
        with self._lock:
            self._resources.append(_Resource("schema", name))
        return name

    def register_table(self, schema: str, name: str) -> TableResource:
        self._validate(schema)
        self._validate(name)
        with self._lock:
            self._resources.append(_Resource("table", name, schema=schema))
        return TableResource(schema=schema, name=name)

    def register_index(self, schema: str, name: str) -> str:
        self._validate(schema)
        self._validate(name)
        with self._lock:
            self._resources.append(_Resource("index", name, schema=schema))
        return name

    def register_session_id(self, name: str) -> str:
        self._validate(name)
        with self._lock:
            self._resources.append(_Resource("session_id", name))
        return name

    def register_session(self, label: str, backend_pid: int) -> int:
        self._validate(label)
        if not isinstance(backend_pid, int) or backend_pid <= 0:
            raise ValueError("backend_pid must be a positive integer")
        rows = self._control.fetch_all(
            CompiledSQL(
                sql.SQL(
                    "SELECT backend_start, application_name "
                    "FROM pg_stat_activity "
                    "WHERE pid = %s AND datname = current_database()"
                ),
                (backend_pid,),
            ),
            operation="register E2E session identity",
        )
        if len(rows) != 1:
            raise ValueError(
                "registered E2E session must match exactly one current database backend"
            )
        backend_start, application_name = rows[0]
        if not isinstance(backend_start, datetime):
            raise ValueError("registered E2E session backend_start is invalid")
        if application_name != self.prefix:
            raise ValueError(
                "registered E2E session application_name is outside this E2E prefix"
            )
        with self._lock:
            self._resources.append(
                _Resource(
                    "session",
                    label,
                    backend_pid=backend_pid,
                    backend_start=backend_start,
                    application_name=application_name,
                )
            )
        return backend_pid

    def release_session(self, label: str, backend_pid: int) -> None:
        self._validate(label)
        with self._lock:
            matches = [
                index
                for index, resource in enumerate(self._resources)
                if resource.kind == "session"
                and resource.name == label
                and resource.backend_pid == backend_pid
            ]
            if len(matches) != 1:
                raise ValueError(
                    "registered E2E session release must match exactly once"
                )
            self._resources.pop(matches[0])

    def register_engine(self, label: str, engine: GaussDBEngine) -> GaussDBEngine:
        self._validate(label)
        with self._lock:
            self._resources.append(_Resource("engine_label", label))
        return engine

    def _drop(self, resource: _Resource) -> None:
        if resource.kind == "session":
            self._control.fetch_all(
                CompiledSQL(
                    sql.SQL(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE pid = %s AND backend_start = %s "
                        "AND application_name = %s "
                        "AND datname = current_database()"
                    ),
                    (
                        resource.backend_pid,
                        resource.backend_start,
                        resource.application_name,
                    ),
                ),
                operation="cleanup registered E2E session",
            )
        elif resource.kind == "index":
            self._control.execute(
                CompiledSQL(
                    sql.SQL("DROP INDEX IF EXISTS {}.{}").format(
                        sql.Identifier(resource.schema),
                        sql.Identifier(resource.name),
                    )
                ),
                operation="cleanup registered E2E index",
            )
        elif resource.kind == "table":
            self._control.execute(
                CompiledSQL(
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier(resource.schema),
                        sql.Identifier(resource.name),
                    )
                ),
                operation="cleanup registered E2E table",
            )
        elif resource.kind == "schema":
            self._control.execute(
                CompiledSQL(
                    sql.SQL("DROP SCHEMA IF EXISTS {}").format(
                        sql.Identifier(resource.name)
                    )
                ),
                operation="cleanup registered E2E schema",
            )
        elif resource.kind == "engine_label":
            return

    def _is_absent(self, resource: _Resource) -> bool:
        if resource.kind == "engine_label":
            return True
        if resource.kind == "session_id":
            return True
        if resource.kind == "session":
            rows = self._control.fetch_all(
                CompiledSQL(
                    sql.SQL(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM pg_stat_activity "
                        "WHERE pid = %s AND backend_start = %s "
                        "AND application_name = %s "
                        "AND datname = current_database())"
                    ),
                    (
                        resource.backend_pid,
                        resource.backend_start,
                        resource.application_name,
                    ),
                ),
                operation="verify E2E session cleanup",
            )
        elif resource.kind == "schema":
            rows = self._control.fetch_all(
                CompiledSQL(
                    sql.SQL(
                        "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)"
                    ),
                    (resource.name,),
                ),
                operation="verify E2E schema cleanup",
            )
        else:
            kind_predicate = (
                "rel.relkind IN ('i', 'I')"
                if resource.kind == "index"
                else "rel.relkind IN ('r', 'p')"
            )
            rows = self._control.fetch_all(
                CompiledSQL(
                    sql.SQL(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM pg_class rel "
                        "JOIN pg_namespace ns ON ns.oid = rel.relnamespace "
                        "WHERE ns.nspname = %s AND rel.relname = %s "
                        f"AND {kind_predicate})"
                    ),
                    (resource.schema, resource.name),
                ),
                operation=f"verify E2E {resource.kind} cleanup",
            )
        return rows == [(False,)]

    def cleanup(self, *, primary_failed: bool = False) -> None:
        failures: list[BaseException] = []
        with self._lock:
            current = list(self._resources)
            self._resources.clear()
        if not current:
            return
        order = {
            "session": 0,
            "session_id": 0,
            "index": 1,
            "table": 2,
            "schema": 3,
            "engine_label": 4,
        }
        resources = sorted(
            reversed(current),
            key=lambda item: order[item.kind],
        )
        remaining: list[_Resource] = []
        for resource in resources:
            try:
                self._drop(resource)
                absent = self._is_absent(resource)
                self.evidence.append(
                    CleanupEvidence(
                        owner_test=self.owner_test,
                        kind=resource.kind,
                        name=resource.name,
                        dropped=True,
                        absent_after_cleanup=absent,
                    )
                )
                if not absent:
                    failures.append(
                        AssertionError(
                            f"registered {resource.kind} was not absent after cleanup"
                        )
                    )
                    remaining.append(resource)
            except BaseException as exc:
                sanitized = sanitize_exception(exc)
                self.evidence.append(
                    CleanupEvidence(
                        owner_test=self.owner_test,
                        kind=resource.kind,
                        name=resource.name,
                        dropped=False,
                        absent_after_cleanup=False,
                        error_class=sanitized["exception_class"],
                        sqlstate=sanitized["sqlstate"],
                    )
                )
                failures.append(exc)
                remaining.append(resource)
        if remaining:
            with self._lock:
                self._resources.extend(remaining)
        if failures:
            message = (
                f"{len(failures)} registered E2E resource(s) failed cleanup "
                f"for {self.owner_test}"
            )
            if primary_failed:
                warnings.warn(message, RuntimeWarning, stacklevel=1)
            else:
                raise AssertionError(message) from None


@pytest.fixture
def e2e_prefix() -> str:
    worker = (
        re.sub(
            r"[^a-z0-9]",
            "",
            os.environ.get("PYTEST_XDIST_WORKER", "main").lower(),
        )
        or "main"
    )
    return f"lcg_e2e_{worker}_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def resource_registry(
    request: pytest.FixtureRequest,
    e2e_prefix: str,
    control_engine: GaussDBEngine,
) -> ResourceRegistry:
    registry = ResourceRegistry(
        e2e_prefix,
        control_engine,
        owner_test=request.node.nodeid.split("[", 1)[0],
    )

    def finalize() -> None:
        report = getattr(request.node, "rep_call", None)
        registry.cleanup(primary_failed=bool(report is not None and report.failed))

    request.addfinalizer(finalize)
    return registry


@pytest.fixture
def e2e_namespace(
    e2e_prefix: str,
    resource_registry: ResourceRegistry,
) -> E2ENamespace:
    return E2ENamespace(prefix=e2e_prefix, registry=resource_registry)


@pytest.fixture
def writable_engine(
    writable_engine_factory: Callable[..., GaussDBEngine],
    e2e_namespace: E2ENamespace,
    resource_registry: ResourceRegistry,
) -> GaussDBEngine:
    engine = writable_engine_factory(minconn=1, maxconn=4)
    return resource_registry.register_engine(
        e2e_namespace.name("engine"),
        engine,
    )


@pytest.fixture
def temporary_schema(
    e2e_namespace: E2ENamespace,
    resource_registry: ResourceRegistry,
    writable_engine: GaussDBEngine,
) -> str:
    name = e2e_namespace.schema()
    created = False
    try:
        writable_engine.execute(
            CompiledSQL(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name))),
            operation="create temporary E2E schema",
        )
        created = True
    finally:
        if not created:
            resource_registry.cleanup(primary_failed=True)
    return name


@pytest.fixture
def temporary_vector_table(
    e2e_namespace: E2ENamespace,
    resource_registry: ResourceRegistry,
    writable_engine: GaussDBEngine,
    temporary_schema: str,
) -> TableResource:
    table = e2e_namespace.table(temporary_schema, "vector")
    created = False
    try:
        writable_engine.execute(
            CompiledSQL(
                sql.SQL(
                    "CREATE TABLE {}.{} ("
                    "id TEXT PRIMARY KEY, content TEXT NOT NULL, "
                    "metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, "
                    "embedding FLOATVECTOR(3) NOT NULL)"
                    " WITH (storage_type=ustore)"
                ).format(
                    sql.Identifier(table.schema),
                    sql.Identifier(table.name),
                )
            ),
            operation="create temporary E2E vector table",
        )
        created = True
    finally:
        if not created:
            resource_registry.cleanup(primary_failed=True)
    return table


@pytest.fixture
def temporary_chat_table(
    e2e_namespace: E2ENamespace,
    resource_registry: ResourceRegistry,
    writable_engine: GaussDBEngine,
    temporary_schema: str,
) -> TableResource:
    table = e2e_namespace.table(temporary_schema, "chat")
    created = False
    try:
        writable_engine.execute(
            CompiledSQL(
                sql.SQL(
                    "CREATE TABLE {}.{} ("
                    "id BIGSERIAL PRIMARY KEY, session_id TEXT NOT NULL, "
                    "message JSONB NOT NULL, "
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                    " WITH (storage_type=ustore)"
                ).format(
                    sql.Identifier(table.schema),
                    sql.Identifier(table.name),
                )
            ),
            operation="create temporary E2E chat table",
        )
        created = True
    finally:
        if not created:
            resource_registry.cleanup(primary_failed=True)
    return table


@pytest.fixture
def vector_table(temporary_vector_table: TableResource) -> TableResource:
    return temporary_vector_table


@pytest.fixture
def chat_table(temporary_chat_table: TableResource) -> TableResource:
    return temporary_chat_table


@pytest.fixture
def capability_snapshot(control_engine: GaussDBEngine) -> dict[str, object]:
    access_methods = {
        str(row[0]).lower()
        for row in control_engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT amname FROM pg_am ORDER BY amname")),
            operation="probe E2E access methods",
        )
    }
    opclasses = {
        str(row[0]).lower()
        for row in control_engine.fetch_all(
            CompiledSQL(sql.SQL("SELECT opcname FROM pg_opclass ORDER BY opcname")),
            operation="probe E2E operator classes",
        )
    }
    bm25_operator = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT EXISTS ("
                "SELECT 1 FROM pg_operator op "
                "WHERE op.oprname = %s "
                "AND op.oprleft = 'text'::regtype "
                "AND op.oprright = 'text'::regtype "
                "AND pg_operator_is_visible(op.oid))"
            ),
            ("###",),
        ),
        operation="probe E2E BM25 operator",
    ) == [(True,)]
    settings = control_engine.fetch_all(
        CompiledSQL(
            sql.SQL(
                "SELECT current_setting('server_version_num'), "
                "current_setting('default_transaction_read_only'), "
                "EXISTS (SELECT 1 FROM pg_roles WHERE rolname = "
                "'gs_role_signal_backend' AND "
                "pg_has_role(current_user, oid, 'MEMBER')), "
                "EXISTS (SELECT 1 FROM pg_catalog.pgxc_node "
                "WHERE node_type = 'D')"
            )
        ),
        operation="probe E2E server capabilities",
    )
    version, read_only, can_terminate, distributed = settings[0]
    return {
        "access_methods": tuple(sorted(access_methods)),
        "opclasses": tuple(sorted(opclasses)),
        "bm25": bm25_operator and not distributed,
        "distributed": bool(distributed),
        "ugin": "ugin" in access_methods,
        "server_version_class": str(version)[:2],
        "default_transaction_read_only": str(read_only).lower(),
        "backend_termination_privilege": bool(can_terminate),
    }


def _require(snapshot: dict[str, object], key: str) -> bool:
    if snapshot.get(key) is not True:
        pytest.skip(f"GaussDB capability {key} is explicitly unavailable")
    return True


@pytest.fixture
def requires_gsdiskann(capability_snapshot: dict[str, object]) -> bool:
    access = capability_snapshot["access_methods"]
    if "gsdiskann" not in access:
        pytest.skip("GaussDB capability gsdiskann is explicitly unavailable")
    return True


@pytest.fixture
def requires_bm25(capability_snapshot: dict[str, object]) -> bool:
    if capability_snapshot.get("distributed") is True:
        pytest.skip(
            "BM25 and hybrid retrieval are not supported on distributed GaussDB"
        )
    return _require(capability_snapshot, "bm25")


@pytest.fixture
def requires_ugin(capability_snapshot: dict[str, object]) -> bool:
    return _require(capability_snapshot, "ugin")


@pytest.fixture
def requires_backend_termination_privilege(
    capability_snapshot: dict[str, object],
) -> bool:
    return _require(capability_snapshot, "backend_termination_privilege")


class DeterministicSyncAsyncEmbeddings(Embeddings):
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls = {
            "embed_documents": 0,
            "aembed_documents": 0,
            "embed_query": 0,
            "aembed_query": 0,
        }
        self._lock = threading.Lock()

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [
            (int.from_bytes(digest[index * 2 : index * 2 + 2], "big") / 65535)
            for index in range(self.dimension)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.calls["embed_documents"] += 1
        return [self._vector(text) for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self.calls["aembed_documents"] += 1
        await asyncio.sleep(0)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            self.calls["embed_query"] += 1
        return self._vector(text)

    async def aembed_query(self, text: str) -> list[float]:
        with self._lock:
            self.calls["aembed_query"] += 1
        await asyncio.sleep(0)
        return self._vector(text)


@pytest.fixture
def deterministic_embeddings() -> DeterministicSyncAsyncEmbeddings:
    return DeterministicSyncAsyncEmbeddings()
