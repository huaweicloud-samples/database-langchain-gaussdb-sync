from langchain_gaussdb.errors import (
    GaussDBConnectionError,
    GaussDBSQLError,
    GaussDBTransactionError,
    sanitize_dsn,
    wrap_psycopg_error,
)


class FakePsycopgError(Exception):
    pgcode = "23505"

    def __str__(self):
        return "duplicate key value violates unique constraint"


class SensitivePsycopgError(Exception):
    pgcode = "08006"

    def __str__(self):
        return "connect failed dsn=host=127.0.0.1 password=raw-secret"


def test_sanitize_keyword_dsn_masks_password():
    dsn = "host=127.0.0.1 port=19995 user=lxm password=secret dbname=mem0"
    sanitized = sanitize_dsn(dsn)
    assert "secret" not in sanitized
    assert "password=***" in sanitized
    assert "user=lxm" in sanitized


def test_sanitize_url_dsn_masks_password():
    dsn = "postgresql://lxm:secret@127.0.0.1:19995/mem0"
    sanitized = sanitize_dsn(dsn)
    assert "secret" not in sanitized
    assert "lxm:***@" in sanitized


def test_sanitize_dsn_masks_api_key_like_values():
    dsn = "host=127.0.0.1 api_key=sk-secret apikey=another-secret"
    sanitized = sanitize_dsn(dsn)
    assert "sk-secret" not in sanitized
    assert "another-secret" not in sanitized
    assert "api_key=***" in sanitized
    assert "apikey=***" in sanitized


def test_sanitize_keyword_dsn_masks_quoted_password_with_spaces():
    dsn = "host=127.0.0.1 password='secret with spaces' user=lxm"
    sanitized = sanitize_dsn(dsn)
    assert "secret" not in sanitized
    assert "spaces" not in sanitized
    assert "password=***" in sanitized
    assert "user=lxm" in sanitized


def test_sanitize_keyword_dsn_masks_unquoted_backslash_escaped_password():
    dsn = r"host=127.0.0.1 password=abc\ def user=lxm"

    sanitized = sanitize_dsn(dsn)

    assert "abc" not in sanitized
    assert " def" not in sanitized
    assert "password=***" in sanitized
    assert "host=127.0.0.1" in sanitized
    assert "user=lxm" in sanitized


def test_sanitize_keyword_dsn_masks_quoted_backslash_escaped_password():
    dsn = r"host=127.0.0.1 password='abc\'def' user=lxm"

    sanitized = sanitize_dsn(dsn)

    assert "abc" not in sanitized
    assert "def'" not in sanitized
    assert "password=***" in sanitized
    assert "host=127.0.0.1" in sanitized
    assert "user=lxm" in sanitized


def test_sanitize_keyword_dsn_masks_ampersand_in_unquoted_password():
    dsn = "host=127.0.0.1 password=alpha&omega user=lxm"

    sanitized = sanitize_dsn(dsn)

    assert "alpha" not in sanitized
    assert "omega" not in sanitized
    assert "password=***" in sanitized
    assert "host=127.0.0.1" in sanitized
    assert "user=lxm" in sanitized


def test_sanitize_keyword_dsn_masks_hash_in_unquoted_password():
    dsn = "host=127.0.0.1 password=alpha#omega user=lxm"

    sanitized = sanitize_dsn(dsn)

    assert "alpha" not in sanitized
    assert "omega" not in sanitized
    assert "password=***" in sanitized
    assert "host=127.0.0.1" in sanitized
    assert "user=lxm" in sanitized


def test_sanitize_url_dsn_masks_password_containing_at_sign():
    dsn = "postgresql://lxm:p@ss@127.0.0.1:19995/mem0"
    sanitized = sanitize_dsn(dsn)
    assert "p@ss" not in sanitized
    assert "ss@" not in sanitized
    assert "lxm:***@127.0.0.1" in sanitized


def test_sanitize_uri_query_password_preserves_following_parameters():
    dsn = "postgresql://host/db?password=query-secret&sslmode=require"

    sanitized = sanitize_dsn(dsn)

    assert "query-secret" not in sanitized
    assert sanitized == "postgresql://host/db?password=***&sslmode=require"


def test_sanitize_uri_query_decodes_partially_encoded_sensitive_key():
    dsn = "postgresql://host/db?source=docs&pass%77ord=query-secret&sslmode=require"

    sanitized = sanitize_dsn(dsn)

    assert "query-secret" not in sanitized
    assert sanitized == (
        "postgresql://host/db?source=docs&pass%77ord=***&sslmode=require"
    )


def test_sanitize_keyword_boundary_does_not_match_hyphenated_setting_name():
    text = "reset-password=visible password=secret"

    sanitized = sanitize_dsn(text)

    assert "reset-password=visible" in sanitized
    assert "password=secret" not in sanitized
    assert sanitized.endswith("password=***")


def test_wrap_psycopg_error_preserves_sqlstate():
    wrapped = wrap_psycopg_error(
        FakePsycopgError(),
        operation="insert documents",
        context={"table": "documents"},
        error_cls=GaussDBSQLError,
    )
    assert isinstance(wrapped, GaussDBSQLError)
    assert wrapped.sqlstate == "23505"
    assert "insert documents" in str(wrapped)
    assert "documents" in str(wrapped)


def test_wrapped_error_does_not_leak_sensitive_dsn():
    wrapped = wrap_psycopg_error(
        FakePsycopgError(),
        operation="connect",
        context={
            "dsn": "host=127.0.0.1 user=lxm password=secret",
            "api_key": "sk-secret",
        },
        error_cls=GaussDBConnectionError,
    )
    message = str(wrapped)
    assert "secret" not in message
    assert "sk-secret" not in message
    assert "password=***" in message
    assert "api_key=***" in message


def test_wrapped_error_redacts_bare_sensitive_context_value_from_cause():
    wrapped = wrap_psycopg_error(
        Exception("bad token sk-secret"),
        operation="connect",
        context={"token": "sk-secret"},
        error_cls=GaussDBConnectionError,
    )
    message = str(wrapped)
    assert "sk-secret" not in message
    assert "token=***" in message


def test_wrapped_error_redacts_bare_dsn_password_value_from_cause():
    wrapped = wrap_psycopg_error(
        Exception("could not connect using secret"),
        operation="connect",
        context={"dsn": "host=127.0.0.1 password=secret"},
        error_cls=GaussDBConnectionError,
    )
    message = str(wrapped)
    assert "secret" not in message
    assert "password=***" in message


def test_wrapped_error_redacts_sensitive_values_copied_into_context_fields():
    wrapped = wrap_psycopg_error(
        Exception("rollback failed"),
        operation="rollback",
        context={
            "token": "sk-secret",
            "dsn": "host=127.0.0.1 password=dsn-secret",
            "primary_failure": "FakeError: bad token sk-secret dsn dsn-secret",
        },
        error_cls=GaussDBTransactionError,
    )

    message = str(wrapped)
    assert "sk-secret" not in message
    assert "dsn-secret" not in message
    assert "token=***" in message
    assert "password=***" in message
    assert "primary_failure=FakeError: bad token *** dsn ***" in message


def test_wrapped_error_does_not_keep_raw_cause_that_may_leak_secret():
    wrapped = wrap_psycopg_error(
        SensitivePsycopgError(),
        operation="connect",
        context={"dsn": "host=127.0.0.1 password=context-secret"},
        error_cls=GaussDBConnectionError,
    )
    assert wrapped.__cause__ is None
    assert "raw-secret" not in str(wrapped)
    assert "context-secret" not in str(wrapped)


def test_wrapped_error_redacts_libpq_credentials_from_all_message_sources():
    wrapped = wrap_psycopg_error(
        Exception("connect failed sslpassword=cause-ssl-secret"),
        operation=(
            "connect postgresql://user:operation-url-secret@127.0.0.1/db "
            "sslpassword=operation-ssl-secret"
        ),
        context={
            "sslpassword": "context-ssl-secret",
            "passfile": "C:/private/context-passfile",
        },
        error_cls=GaussDBConnectionError,
    )

    message = str(wrapped)
    for secret in (
        "cause-ssl-secret",
        "operation-url-secret",
        "operation-ssl-secret",
        "context-ssl-secret",
        "context-passfile",
    ):
        assert secret not in message
    assert "sslpassword=***" in message
    assert "passfile=***" in message


def test_wrapped_error_redacts_backslash_escaped_passwords_without_losing_context():
    wrapped = wrap_psycopg_error(
        Exception(
            r"authentication rejected password='causealpha\'causeomega' "
            "for bare ctxalpha ctxomega"
        ),
        operation=r"connect password=opalpha\ opomega table=docs",
        context={
            "dsn": r"host=127.0.0.1 password=ctxalpha\ ctxomega user=lxm",
            "table": "docs",
        },
        error_cls=GaussDBConnectionError,
    )

    message = str(wrapped)
    for secret_fragment in (
        "opalpha",
        "opomega",
        "causealpha",
        "causeomega",
        "ctxalpha",
        "ctxomega",
    ):
        assert secret_fragment not in message
    assert "password=***" in message
    assert "host=127.0.0.1" in message
    assert "user=lxm" in message
    assert "table=docs" in message
    assert "authentication rejected" in message


def test_wrapped_error_redacts_uri_query_password_and_bare_context_secret():
    wrapped = wrap_psycopg_error(
        Exception("authentication rejected for query-secret"),
        operation="connect",
        context={
            "dsn": ("postgresql://host/db?password=query-secret&sslmode=require"),
            "table": "docs",
        },
        error_cls=GaussDBConnectionError,
    )

    message = str(wrapped)
    assert "query-secret" not in message
    assert "password=***&sslmode=require" in message
    assert "authentication rejected" in message
    assert "table=docs" in message


def test_wrapped_error_redacts_decoded_percent_encoded_uri_query_password():
    wrapped = wrap_psycopg_error(
        Exception("authentication rejected for s3cret-value"),
        operation=(
            "connect postgresql://host/db?password=s3cr%65t%2Dvalue"
            "&application_name=langchain"
        ),
        context={
            "dsn": ("postgresql://host/db?password=s3cr%65t%2Dvalue&sslmode=require")
        },
        error_cls=GaussDBConnectionError,
    )

    message = str(wrapped)
    assert "s3cr%65t%2Dvalue" not in message
    assert "s3cret-value" not in message
    assert "password=***&application_name=langchain" in message
    assert "password=***&sslmode=require" in message


def test_wrapped_error_decodes_fully_encoded_sensitive_query_key():
    wrapped = wrap_psycopg_error(
        Exception("authentication rejected for encoded-key-secret"),
        operation="connect",
        context={
            "dsn": (
                "postgresql://host/db?%70assword=encoded-key-secret&sslmode=require"
            )
        },
        error_cls=GaussDBConnectionError,
    )

    message = str(wrapped)
    assert "encoded-key-secret" not in message
    assert "%70assword=***&sslmode=require" in message
    assert "authentication rejected" in message
