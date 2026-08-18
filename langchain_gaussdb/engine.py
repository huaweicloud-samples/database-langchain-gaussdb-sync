from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn, TypeVar

import psycopg2
from psycopg2 import sql
from psycopg2.pool import ThreadedConnectionPool

from langchain_gaussdb.errors import (
    GaussDBConnectionError,
    GaussDBSQLError,
    GaussDBTransactionError,
    _redact_sensitive_context_values,
    sanitize_dsn,
    wrap_psycopg_error,
)
from langchain_gaussdb.sql import CompiledSQL

ResultT = TypeVar("ResultT")


@dataclass
class _CloseAttempt:
    done: bool = False
    error: BaseException | None = None


class GaussDBEngine:
    """Thread-safe synchronous execution core for GaussDB operations."""

    def __init__(
        self,
        *,
        dsn: str | None = None,
        connection_kwargs: Mapping[str, Any] | None = None,
        minconn: int = 1,
        maxconn: int = 10,
    ) -> None:
        if (dsn is None) == (connection_kwargs is None):
            raise GaussDBConnectionError(
                "GaussDBEngine requires exactly one connection source"
            )
        if minconn < 1 or maxconn < minconn:
            raise GaussDBConnectionError(
                "GaussDBEngine requires 1 <= minconn <= maxconn"
            )
        if connection_kwargs is not None and "async_" in connection_kwargs:
            raise GaussDBConnectionError(
                "GaussDBEngine does not support the async_ connection option"
            )

        if dsn is not None:
            self._connection_context: dict[str, Any] = {"dsn": dsn}
            self._pool = _create_threaded_pool(
                minconn,
                maxconn,
                context=self._connection_context,
                dsn=dsn,
            )
        else:
            connection_options = dict(connection_kwargs or {})
            self._connection_context = connection_options
            self._pool = _create_threaded_pool(
                minconn,
                maxconn,
                context=self._connection_context,
                connection_kwargs=connection_options,
            )

        # ThreadedConnectionPool raises immediately when exhausted.  The
        # semaphore turns that behavior into bounded waiting, which is needed
        # when LangChain's async fallbacks fan out work across executor threads.
        self._pool_semaphore = threading.Semaphore(maxconn)
        self._lifecycle = threading.Condition(threading.RLock())
        self._active_operations = 0
        self._closing = False
        self._closed = False
        self._poisoned = False
        self._close_attempt: _CloseAttempt | None = None
        self._discarded_connections: set[int] = set()
        self._operation_local = threading.local()

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Borrow a raw synchronous connection for the duration of the context."""

        self._enter_operation()
        semaphore_acquired = False
        connection: Any | None = None
        body_error: BaseException | None = None
        try:
            try:
                self._pool_semaphore.acquire()
                semaphore_acquired = True
                connection = self._pool.getconn()
            except BaseException as exc:
                raise self._connection_error(
                    exc,
                    operation="borrow connection",
                ) from None

            try:
                yield connection
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                if not self._consume_discarded_connection(connection):
                    recovery_error: BaseException | None = None
                    try:
                        reusable = self._prepare_connection_for_return(connection)
                    except BaseException as exc:
                        recovery_error = self._sanitize_connection_exception(exc)
                        reusable = False

                    pool_error: BaseException | None = None
                    try:
                        self._pool.putconn(connection, close=not reusable)
                    except BaseException as exc:
                        sanitized_pool_error = self._sanitize_connection_exception(exc)
                        with self._lifecycle:
                            self._poisoned = True
                        self._close_failed_connection(connection)
                        primary_error = (
                            recovery_error if recovery_error is not None else body_error
                        )
                        context = _with_primary_failure_context(None, primary_error)
                        if recovery_error is not None and body_error is not None:
                            context["body_failure"] = sanitize_dsn(str(body_error))
                        pool_error = self._connection_error(
                            sanitized_pool_error,
                            operation=(
                                "return connection"
                                if reusable
                                else "discard connection"
                            ),
                            context=context,
                        )
                        if recovery_error is not None:
                            _add_sanitized_cleanup_note(
                                pool_error,
                                recovery_error,
                                operation="connection recovery primary",
                            )

                    if pool_error is not None:
                        _raise_without_exception_context(pool_error)

                    if recovery_error is not None:
                        if body_error is None:
                            raise recovery_error
                        _add_sanitized_cleanup_note(
                            body_error,
                            recovery_error,
                            operation="connection recovery",
                        )
        finally:
            if semaphore_acquired:
                self._pool_semaphore.release()
            self._exit_operation()

    def transaction(
        self,
        callback: Callable[[Any], ResultT],
        *,
        operation: str = "transaction",
        context: Mapping[str, Any] | None = None,
    ) -> ResultT:
        """Run a callback in one synchronous database transaction."""

        with self.connection() as connection:
            cursor: Any | None = None
            try:
                try:
                    cursor = connection.cursor()
                except BaseException as cursor_error:
                    try:
                        connection.rollback()
                    except BaseException as rollback_error:
                        self._discard_failed_connection(connection)
                        raise self._cleanup_error(
                            rollback_error,
                            primary_error=cursor_error,
                            operation=f"{operation} rollback after cursor failure",
                            context=context,
                        ) from None
                    self._discard_failed_connection(connection)
                    raise self._transaction_error(
                        cursor_error,
                        operation=f"{operation} cursor",
                        context=context,
                    ) from None

                try:
                    result = callback(cursor)
                except BaseException as primary_error:
                    try:
                        connection.rollback()
                    except BaseException as rollback_error:
                        self._discard_failed_connection(connection)
                        raise self._cleanup_error(
                            rollback_error,
                            primary_error=primary_error,
                            operation=f"{operation} rollback",
                            context=context,
                        ) from None

                    if _is_psycopg_error(primary_error):
                        raise wrap_psycopg_error(
                            primary_error,
                            operation=operation,
                            context=context,
                            error_cls=GaussDBSQLError,
                        ) from None
                    if isinstance(primary_error, Exception):
                        raise
                    raise _copy_sanitized_base_exception(primary_error) from None

                try:
                    connection.commit()
                except BaseException as commit_error:
                    try:
                        connection.rollback()
                    except BaseException as rollback_error:
                        self._discard_failed_connection(connection)
                        raise self._cleanup_error(
                            rollback_error,
                            primary_error=commit_error,
                            operation=f"{operation} rollback after commit failure",
                            context=context,
                        ) from None
                    raise self._transaction_error(
                        commit_error,
                        operation=f"{operation} commit",
                        context=context,
                    ) from None

                return result
            finally:
                if cursor is not None:
                    active_error = sys.exc_info()[1]
                    try:
                        cursor.close()
                    except BaseException as close_error:
                        if active_error is None:
                            raise self._transaction_error(
                                close_error,
                                operation=f"{operation} cursor close",
                                context=context,
                            ) from None

    def execute(
        self,
        compiled: CompiledSQL,
        *,
        operation: str = "execute",
    ) -> None:
        def execute_compiled(cursor: Any) -> None:
            cursor.execute(compiled.statement, compiled.params)

        self.transaction(execute_compiled, operation=operation)

    def fetch_all(
        self,
        compiled: CompiledSQL,
        *,
        operation: str = "fetch_all",
    ) -> list[tuple[Any, ...]]:
        def fetch(cursor: Any) -> list[tuple[Any, ...]]:
            cursor.execute(compiled.statement, compiled.params)
            return list(cursor.fetchall())

        return self.transaction(fetch, operation=operation)

    def check_connection(self) -> bool:
        rows = self.fetch_all(
            CompiledSQL(sql.SQL("SELECT %s"), [1]),
            operation="check connection",
        )
        return bool(rows)

    def close(self) -> None:
        """Wait for active work and close the owned pool exactly once."""

        if self._operation_depth():
            raise GaussDBConnectionError(
                "GaussDBEngine cannot close from an active operation"
            )

        with self._lifecycle:
            if self._closed:
                return
            if self._closing:
                attempt = self._close_attempt
                while attempt is not None and not attempt.done:
                    self._lifecycle.wait()
                if attempt is not None and attempt.error is not None:
                    raise _copy_close_error(attempt.error) from None
                return

            attempt = _CloseAttempt()
            self._close_attempt = attempt
            self._closing = True
            while self._active_operations:
                self._lifecycle.wait()

        close_error: BaseException | None = None
        try:
            self._pool.closeall()
        except BaseException as exc:
            close_error = self._connection_error(exc, operation="close engine")
        finally:
            with self._lifecycle:
                attempt.error = close_error
                attempt.done = True
                self._closed = close_error is None
                self._closing = False
                self._lifecycle.notify_all()

        if close_error is not None:
            raise close_error from None

    def _enter_operation(self) -> None:
        with self._lifecycle:
            if self._closed:
                raise GaussDBConnectionError("GaussDBEngine is closed")
            if self._closing:
                raise GaussDBConnectionError("GaussDBEngine is closing")
            if self._poisoned:
                raise GaussDBConnectionError(
                    "GaussDBEngine is unusable after a connection pool failure"
                )
            if self._operation_depth():
                raise GaussDBConnectionError(
                    "GaussDBEngine does not support same-thread reentrant operations"
                )
            self._active_operations += 1
            self._operation_local.depth = 1

    def _exit_operation(self) -> None:
        with self._lifecycle:
            self._operation_local.depth = 0
            self._active_operations -= 1
            if self._active_operations == 0:
                self._lifecycle.notify_all()

    def _operation_depth(self) -> int:
        return int(getattr(self._operation_local, "depth", 0))

    def _prepare_connection_for_return(self, connection: Any) -> bool:
        """Return whether a borrowed connection is safe to return for reuse."""

        if getattr(connection, "closed", False):
            return False
        transaction_status = getattr(connection, "get_transaction_status", None)
        if not callable(transaction_status):
            return True
        status = transaction_status()
        if status == psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN:
            return False
        if status != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
            connection.rollback()
        return True

    def _discard_failed_connection(self, connection: Any) -> None:
        with self._lifecycle:
            self._discarded_connections.add(id(connection))
        discard_error: BaseException | None = None
        try:
            self._pool.putconn(connection, close=True)
        except BaseException as exc:
            discard_error = self._sanitize_connection_exception(exc)
            with self._lifecycle:
                self._poisoned = True
            self._close_failed_connection(connection)
        if discard_error is not None:
            _raise_without_exception_context(discard_error)

    def _consume_discarded_connection(self, connection: Any) -> bool:
        with self._lifecycle:
            connection_id = id(connection)
            if connection_id not in self._discarded_connections:
                return False
            self._discarded_connections.remove(connection_id)
            return True

    @staticmethod
    def _close_failed_connection(connection: Any) -> None:
        close = getattr(connection, "close", None)
        if not callable(close):
            return
        try:
            close()
        except BaseException:
            pass

    def _connection_error(
        self,
        error: BaseException,
        *,
        operation: str,
        context: Mapping[str, Any] | None = None,
    ) -> BaseException:
        if isinstance(error, GaussDBConnectionError):
            return error
        if isinstance(error, Exception):
            error_context = dict(self._connection_context)
            error_context.update(context or {})
            return wrap_psycopg_error(
                error,
                operation=operation,
                context=error_context,
                error_cls=GaussDBConnectionError,
            )
        return _copy_sanitized_base_exception(error)

    def _sanitize_connection_exception(
        self,
        error: BaseException,
    ) -> BaseException:
        """Redact connection secrets from an exception and its public notes."""

        message = _redact_sensitive_context_values(
            sanitize_dsn(str(error)),
            self._connection_context,
        )
        sanitized_error = _copy_base_exception_with_message(error, message)

        notes = getattr(error, "__notes__", None)
        if notes:
            sanitized_notes = [
                _redact_sensitive_context_values(
                    sanitize_dsn(str(note)),
                    self._connection_context,
                )
                for note in notes
            ]
            setattr(sanitized_error, "__notes__", sanitized_notes)

        sanitized_error.__context__ = None
        sanitized_error.__cause__ = None
        sanitized_error.__suppress_context__ = True
        return sanitized_error

    @staticmethod
    def _transaction_error(
        error: BaseException,
        *,
        operation: str,
        context: Mapping[str, Any] | None,
    ) -> BaseException:
        if isinstance(error, Exception):
            return wrap_psycopg_error(
                error,
                operation=operation,
                context=context,
                error_cls=GaussDBTransactionError,
            )
        return _copy_sanitized_base_exception(error)

    @staticmethod
    def _cleanup_error(
        error: BaseException,
        *,
        primary_error: BaseException,
        operation: str,
        context: Mapping[str, Any] | None,
    ) -> BaseException:
        cleanup_context = _with_primary_failure_context(context, primary_error)
        if isinstance(error, Exception):
            return wrap_psycopg_error(
                error,
                operation=operation,
                context=cleanup_context,
                error_cls=GaussDBTransactionError,
            )
        return _copy_sanitized_base_exception(error)


def _create_threaded_pool(
    minconn: int,
    maxconn: int,
    *,
    context: Mapping[str, Any],
    dsn: str | None = None,
    connection_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    try:
        if dsn is not None:
            return ThreadedConnectionPool(minconn, maxconn, dsn)
        return ThreadedConnectionPool(
            minconn,
            maxconn,
            **dict(connection_kwargs or {}),
        )
    except BaseException as exc:
        if isinstance(exc, Exception):
            raise wrap_psycopg_error(
                exc,
                operation="connect",
                context=context,
                error_cls=GaussDBConnectionError,
            ) from None
        raise _copy_sanitized_base_exception(exc) from None


def _is_psycopg_error(error: BaseException) -> bool:
    return isinstance(error, psycopg2.Error) or hasattr(error, "pgcode")


def _with_primary_failure_context(
    context: Mapping[str, Any] | None,
    primary_error: BaseException | None,
) -> dict[str, Any]:
    merged = dict(context or {})
    if primary_error is not None:
        merged["primary_failure"] = sanitize_dsn(str(primary_error))
    return merged


def _copy_sanitized_base_exception(error: BaseException) -> BaseException:
    message = sanitize_dsn(str(error))
    return _copy_base_exception_with_message(error, message)


def _copy_base_exception_with_message(
    error: BaseException,
    message: str,
) -> BaseException:
    if type(error) is KeyboardInterrupt:
        return KeyboardInterrupt(message)
    if type(error) is SystemExit:
        return SystemExit(message)
    try:
        return error.__class__(message)
    except BaseException:
        if isinstance(error, KeyboardInterrupt):
            return KeyboardInterrupt(message)
        if isinstance(error, SystemExit):
            return SystemExit(message)
        return BaseException(message)


def _raise_without_exception_context(error: BaseException) -> NoReturn:
    try:
        raise error from None
    except BaseException as raised:
        raised.__context__ = None
        raised.__cause__ = None
        raised.__suppress_context__ = True
        raise


def _copy_close_error(error: BaseException) -> BaseException:
    if isinstance(error, GaussDBConnectionError):
        return GaussDBConnectionError(str(error), sqlstate=error.sqlstate)
    return _copy_sanitized_base_exception(error)


def _add_sanitized_cleanup_note(
    error: BaseException,
    cleanup_error: BaseException,
    *,
    operation: str,
) -> None:
    note = f"GaussDBEngine {operation} failed: {sanitize_dsn(str(cleanup_error))}"
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    setattr(error, "__notes__", notes)
