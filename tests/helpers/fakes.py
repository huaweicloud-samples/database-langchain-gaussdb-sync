import socket
import threading
from collections import deque

import psycopg2
from psycopg2 import sql


class FakeCursor:
    def __init__(self, rows=None, fail_on_execute=None, fail_on_close=None):
        self.rows = [] if rows is None else rows
        self.fail_on_execute = fail_on_execute
        self.fail_on_close = fail_on_close
        self.closed = False
        self.executed = []

    def execute(self, statement, params=None):
        if self.fail_on_execute is not None:
            raise self.fail_on_execute
        self.executed.append((statement, tuple(params or ())))

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self):
        if self.fail_on_close is not None:
            raise self.fail_on_close
        self.closed = True


class FakeConnection:
    def __init__(self, cursor=None, fail_commit=None, fail_rollback=None):
        self.cursor_obj = cursor if cursor is not None else FakeCursor()
        self.fail_commit = fail_commit
        self.fail_rollback = fail_rollback
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        if self.fail_commit is not None:
            raise self.fail_commit
        self.commits += 1

    def rollback(self):
        if self.fail_rollback is not None:
            raise self.fail_rollback
        self.rollbacks += 1

    def close(self):
        self.closed = True


class FakeThreadedPool:
    def __init__(self, connections=None):
        self.connections = deque(
            [FakeConnection()] if connections is None else connections
        )
        self.borrowed = []
        self.returned = []
        self.discarded = []
        self.closed = False

    def getconn(self):
        conn = self.connections.popleft() if self.connections else FakeConnection()
        self.borrowed.append(conn)
        return conn

    def putconn(self, conn, key=None, close=False):
        if close:
            conn.close()
            self.discarded.append(conn)
            return
        self.returned.append(conn)
        self.connections.append(conn)

    def closeall(self):
        self.closed = True


class FakeSimplePool(FakeThreadedPool):
    pass


def _statement_text(statement):
    if isinstance(statement, sql.SQL):
        return statement.string
    return str(statement)


class FakeAsyncCursor:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False
        self._rows = []
        self.rowcount = -1
        self.description = None

    def execute(self, statement, params=None):
        text = _statement_text(statement)
        values = tuple(params or ())
        self.connection.executed.append((text, values))
        failure = self.connection.pop_failure(text)
        if failure is not None:
            raise failure
        self.connection.start_statement(text)
        self._rows = list(self.connection.rows_by_sql.get(text, self.connection.rows))
        self.rowcount = len(self._rows)
        self.description = self.connection.description

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def close(self):
        self.closed = True


class FakeAsyncConnection:
    def __init__(self, *, rows=None):
        self.rows = [("ok",)] if rows is None else list(rows)
        self.rows_by_sql = {}
        self.description = None
        self.executed = []
        self.cursors = []
        self.failures = {}
        self.closed = False
        self.cancel_calls = 0
        self.poll_calls = 0
        self.query_started = threading.Event()
        self.blocked_statements = set()
        self.cancel_fails = False
        self.close_failure = None
        self._query_pending = False
        self._cancel_error_pending = False
        self._reader, self._writer = socket.socketpair()

    def cursor(self):
        cursor = FakeAsyncCursor(self)
        self.cursors.append(cursor)
        return cursor

    def poll(self):
        self.poll_calls += 1
        if self._cancel_error_pending:
            self._cancel_error_pending = False
            raise psycopg2.extensions.QueryCanceledError("query cancelled")
        if self._query_pending:
            return psycopg2.extensions.POLL_READ
        return 0

    def fileno(self):
        return self._reader.fileno()

    def cancel(self):
        self.cancel_calls += 1
        if self.cancel_fails:
            raise RuntimeError("cancel failed")
        self._query_pending = False
        self._cancel_error_pending = True
        self._writer.send(b"x")

    def close(self):
        if self.close_failure is not None:
            failure = self.close_failure
            self.close_failure = None
            raise failure
        self.closed = True
        self._reader.close()
        self._writer.close()

    def block_statement(self, statement):
        self.blocked_statements.add(statement)

    def start_statement(self, statement):
        if statement in self.blocked_statements:
            self._query_pending = True
            self.query_started.set()

    def fail_next(self, statement, error):
        self.failures.setdefault(statement, deque()).append(error)

    def pop_failure(self, statement):
        failures = self.failures.get(statement)
        if not failures:
            return None
        return failures.popleft()


class FakeAsyncConnectFactory:
    def __init__(self, connections=None):
        self._connections = deque(connections or ())
        self.calls = []
        self.connections = []
        self.failure = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.failure is not None:
            raise self.failure
        connection = (
            self._connections.popleft() if self._connections else FakeAsyncConnection()
        )
        self.connections.append(connection)
        return connection
