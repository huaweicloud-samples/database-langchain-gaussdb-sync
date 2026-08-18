from __future__ import annotations

import importlib
import inspect
import re
import shutil
import subprocess
import sys
import zipfile
from importlib import metadata
from pathlib import Path
from typing import get_type_hints

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import langchain_gaussdb as root
from langchain_gaussdb import (
    chat_message_history,
    engine,
    errors,
    hybrid_search,
    sql,
    vectorstore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PUBLIC_API = [
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


def _readme_section(readme_text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        readme_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, heading
    return match.group(1)


def test_root_exports_expected_public_api() -> None:
    assert root.__all__ == EXPECTED_PUBLIC_API
    for name in EXPECTED_PUBLIC_API:
        assert hasattr(root, name)


def test_root_exports_are_same_objects_as_modules() -> None:
    expected_objects = {
        "BM25Config": hybrid_search.BM25Config,
        "CompiledSQL": sql.CompiledSQL,
        "GaussDBEngine": engine.GaussDBEngine,
        "GaussDBVectorStore": vectorstore.GaussDBVectorStore,
        "GaussDBChatMessageHistory": chat_message_history.GaussDBChatMessageHistory,
        "GaussDBError": errors.GaussDBError,
        "GaussDBConnectionError": errors.GaussDBConnectionError,
        "GaussDBSQLError": errors.GaussDBSQLError,
        "GaussDBSQLBuildError": errors.GaussDBSQLBuildError,
        "GaussDBTransactionError": errors.GaussDBTransactionError,
        "GaussDBCapabilityError": errors.GaussDBCapabilityError,
        "GaussDBFilterError": errors.GaussDBFilterError,
    }

    for name, module_object in expected_objects.items():
        assert getattr(root, name) is module_object


def test_langchain_async_method_signatures_and_public_surface() -> None:
    vectorstore_async_methods = {
        "aadd_texts",
        "aadd_documents",
        "adelete",
        "aget_by_ids",
        "asimilarity_search",
        "asimilarity_search_by_vector",
        "asimilarity_search_with_score",
        "asimilarity_search_with_relevance_scores",
        "amax_marginal_relevance_search",
        "amax_marginal_relevance_search_by_vector",
        "afrom_texts",
        "afrom_documents",
    }
    declared_public_async = {
        name
        for name, value in vectorstore.GaussDBVectorStore.__dict__.items()
        if not name.startswith("_")
        and inspect.iscoroutinefunction(
            value.__func__ if isinstance(value, classmethod) else value
        )
    }
    assert declared_public_async == set()
    for name in vectorstore_async_methods | {"asearch"}:
        assert name not in vectorstore.GaussDBVectorStore.__dict__
        assert inspect.getattr_static(
            vectorstore.GaussDBVectorStore, name
        ) is inspect.getattr_static(VectorStore, name)

    aadd_hints = get_type_hints(vectorstore.GaussDBVectorStore.aadd_documents)
    afrom_hints = get_type_hints(vectorstore.GaussDBVectorStore.afrom_documents)
    assert aadd_hints["documents"] == list[Document]
    assert afrom_hints["documents"] == list[Document]

    chat_async_methods = {
        name
        for name, value in chat_message_history.GaussDBChatMessageHistory.__dict__.items()
        if not name.startswith("_") and inspect.iscoroutinefunction(value)
    }
    assert chat_async_methods == set()
    for name in {"aget_messages", "aadd_messages", "aclear"}:
        assert name not in chat_message_history.GaussDBChatMessageHistory.__dict__
        assert inspect.getattr_static(
            chat_message_history.GaussDBChatMessageHistory, name
        ) is inspect.getattr_static(BaseChatMessageHistory, name)

    engine_public_async = {
        name
        for name, value in engine.GaussDBEngine.__dict__.items()
        if not name.startswith("_") and inspect.iscoroutinefunction(value)
    }
    assert engine_public_async == set()
    assert not hasattr(vectorstore.GaussDBVectorStore, "asetup")
    assert not hasattr(vectorstore.GaussDBVectorStore, "aclose")
    assert not hasattr(
        chat_message_history.GaussDBChatMessageHistory,
        "aclose",
    )


def test_root_exports_do_not_leak_internal_helpers() -> None:
    internal_names = {
        "compile_metadata_filter",
        "CompiledFilter",
        "build_create_vector_index",
        "build_create_metadata_index",
        "build_create_bm25_index",
        "build_bm25_search_sql",
        "fuse_hybrid_results",
        "HybridFusionConfig",
        "FakeEmbeddings",
    }

    assert internal_names.isdisjoint(root.__all__)
    for name in internal_names:
        assert not hasattr(root, name)


def test_version_is_string() -> None:
    assert isinstance(root.__version__, str)


def test_version_falls_back_to_empty_string_when_package_metadata_missing(
    monkeypatch,
) -> None:
    calls = []

    def raise_package_not_found(package_name: str) -> str:
        calls.append(package_name)
        raise metadata.PackageNotFoundError(package_name)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(metadata, "version", raise_package_not_found)

            reloaded = importlib.reload(root)

            assert reloaded is root
            assert calls == ["database-langchain-gaussdb-sync"]
            assert root.__version__ == ""
    finally:
        importlib.reload(root)


def test_pyproject_metadata_is_parseable_and_scoped_to_subproject() -> None:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    assert pyproject_path.is_file()
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"

    project = pyproject["project"]
    assert project["name"] == "database-langchain-gaussdb-sync"
    assert project["version"] == "0.1.0"
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.10,<3.15"
    assert any(dep == "langchain-core>=1.1.0,<2.0.0" for dep in project["dependencies"])
    assert any(dep == "numpy>=1.26,<3.0" for dep in project["dependencies"])
    assert any(dep == "psycopg2-binary>=2.9.9,<3.0" for dep in project["dependencies"])

    test_dependencies = project["optional-dependencies"]["test"]
    assert any(dep.startswith("pytest") for dep in test_dependencies)
    assert any(dep.startswith("pytest-asyncio") for dep in test_dependencies)
    assert "langchain-tests==1.1.9" in test_dependencies

    package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert package_finder["include"] == ["langchain_gaussdb*"]
    for excluded in (
        "tests*",
        "sr-design*",
        "docs*",
        "scripts*",
        "_official*",
        "_tmp*",
    ):
        assert excluded in package_finder["exclude"]

    package_data = pyproject["tool"]["setuptools"]["package-data"]
    assert package_data["langchain_gaussdb"] == ["py.typed"]


def test_pyproject_does_not_reference_parent_readme_or_legacy_project() -> None:
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    forbidden_fragments = {
        "../pyproject.toml",
        "../README.md",
        "codex_langchain_gaussdb.md",
        "chat_message_histories",
        "vectorstores",
    }

    for fragment in forbidden_fragments:
        assert fragment not in pyproject_text


def test_py_typed_marker_exists() -> None:
    assert (PROJECT_ROOT / "langchain_gaussdb" / "py.typed").is_file()


def test_readme_documents_user_entrypoints_without_secrets() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    required_fragments = {
        'python -m pip install -e ".[test]"',
        "python -m pip install database-langchain-gaussdb-sync",
        "langchain-core",
        "psycopg2",
        "numpy>=1.26,<3.0",
        '$env:GAUSSDB_TEST_DSN = "host=<host> port=<port> dbname=<database> user=<user>"',
        "GaussDBVectorStore",
        "GaussDBChatMessageHistory",
        "GaussDBEngine",
        "BM25Config",
        "as_retriever",
        "setup()",
        "filter={",
        "metadata_indexes",
        "$exists",
        "GsDiskANN",
        "CREATE INDEX IF NOT EXISTS",
        'retrieval_mode="bm25"',
        'retrieval_mode="hybrid"',
        "text_lemmatized_values",
        "bm25_query",
        "GAUSSDB_TEST_DSN",
        "pytest -m gaussdb_e2e",
        "Dense distance scores are lower-is-better.",
        "BM25 scores are higher-is-better.",
        "automatic initialization",
        "delete(ids=None)",
    }
    for fragment in required_fragments:
        assert fragment in readme_text

    assert "psycopg2 2.x" not in readme_text
    assert "psycopg2-binary" not in readme_text
    assert "langchain-core>=" not in readme_text

    forbidden_patterns = [
        r"121\.37\.\d+\.\d+",
        "G" + r"auss_234",
        "s" + r"k-[A-Za-z0-9_-]{12,}",
        r"password\s*=",
        r"(?i)api[_ -]?key\s*[:=]",
        r"[a-z][a-z0-9+.-]*://[^\s:@]+:[^\s:@]+@",
        r"\bmetadata_key\s*=",
        r"\bcolumn_name\s*=",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, readme_text) is None


def test_chinese_readme_is_linked_detailed_and_secret_free() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    chinese_readme = (PROJECT_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    assert "[简体中文](README.zh-CN.md)" in readme_text
    required_fragments = {
        "用户手册",
        "https://github.com/lilee-LI/database-langchain-gaussdb-sync",
        "python -m pip install database-langchain-gaussdb-sync",
        "GaussDBEngine",
        "GaussDBVectorStore",
        "GaussDBChatMessageHistory",
        "store.setup()",
        "CREATE INDEX IF NOT EXISTS",
        "ON DUPLICATE KEY UPDATE",
        'retrieval_mode="dense"',
        'retrieval_mode="bm25"',
        'retrieval_mode="hybrid"',
        "metadata_indexes",
        "$exists",
        "delete(ids=None)",
        "executor 工作线程",
        "E2E 场景按真实数据库部署形态执行",
        "上线检查清单",
        "公共 API 总表",
    }
    for fragment in required_fragments:
        assert fragment in chinese_readme

    assert "huaweicloud-samples" not in chinese_readme
    assert "psycopg2 2.x" not in chinese_readme
    assert "psycopg2-binary" not in chinese_readme
    assert "langchain-core>=" not in chinese_readme

    forbidden_patterns = [
        r"121\.37\.\d+\.\d+",
        "G" + r"auss_234",
        "s" + r"k-[A-Za-z0-9_-]{12,}",
        r"password\s*=",
        r"(?i)api[_ -]?key\s*[:=]",
        r"[a-z][a-z0-9+.-]*://[^\s:@]+:[^\s:@]+@",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, chinese_readme) is None


def test_readme_vectorstore_documents_automatic_and_explicit_initialization() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    vectorstore_section = _readme_section(readme_text, "VectorStore")

    assert "store.setup()" in vectorstore_section
    assert "create_table_if_not_exists" not in vectorstore_section


def test_readme_bm25_and_hybrid_examples_use_standard_retriever() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    retrieval_section = _readme_section(readme_text, "BM25 And Hybrid Retrieval")

    assert 'retrieval_mode="bm25"' in retrieval_section
    assert 'retrieval_mode="hybrid"' in retrieval_section
    assert ".as_retriever(" in retrieval_section
    assert "as_bm25_retriever" not in retrieval_section
    assert "as_hybrid_retriever" not in retrieval_section
    assert ".setup()" not in retrieval_section


def test_readme_mentions_async_and_delete_all_boundaries() -> None:
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    lower_readme = readme_text.lower()

    assert "standard async compatibility" in lower_readme
    assert "executor worker" in lower_readme
    assert "driver-native async" in lower_readme
    assert "threadedconnectionpool" in lower_readme
    assert "psycopg2" in lower_readme
    assert "delete_all=True" in readme_text
    assert "delete(ids=None)" in readme_text
    assert "does not delete every row by default" in lower_readme
    assert "langchain vectorstore contract" in lower_readme
    assert "does not guarantee server-side query" in lower_readme
    assert "may finish in its worker" in lower_readme


def build_wheel(tmp_path: Path) -> set[str]:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    workspace_build_artifacts = [
        PROJECT_ROOT / "build",
        PROJECT_ROOT / "dist",
        PROJECT_ROOT / "database_langchain_gaussdb_sync.egg-info",
    ]
    preexisting_artifacts = {
        artifact for artifact in workspace_build_artifacts if artifact.exists()
    }

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "-w",
                str(wheelhouse),
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout

        wheels = sorted(wheelhouse.glob("*.whl"))
        assert len(wheels) == 1, [wheel.name for wheel in wheels]
        assert "database_langchain_gaussdb_sync" in wheels[0].name

        with zipfile.ZipFile(wheels[0]) as wheel:
            return set(wheel.namelist())
    finally:
        for artifact in workspace_build_artifacts:
            if artifact in preexisting_artifacts or not artifact.exists():
                continue
            if artifact.is_dir():
                shutil.rmtree(artifact)
            else:
                artifact.unlink()


def test_wheel_build_contains_package_files(tmp_path: Path) -> None:
    wheel_names = build_wheel(tmp_path)

    expected_package_files = {
        "langchain_gaussdb/__init__.py",
        "langchain_gaussdb/vectorstore.py",
        "langchain_gaussdb/chat_message_history.py",
        "langchain_gaussdb/py.typed",
    }

    assert expected_package_files.issubset(wheel_names)


def test_wheel_build_excludes_design_tests_and_parent_legacy_files(
    tmp_path: Path,
) -> None:
    wheel_names = build_wheel(tmp_path)

    forbidden_exact_files = {
        "codex_langchain_gaussdb.md",
        "langchain_gaussdb/_async_driver.py",
    }
    forbidden_prefixes = (
        "tests/",
        "sr-design/",
        "langchain_gaussdb/chat_message_histories/",
        "_official/",
    )

    assert wheel_names.isdisjoint(forbidden_exact_files)
    assert not any(name.startswith(forbidden_prefixes) for name in wheel_names)
    assert not any(
        part.startswith("_tmp") for name in wheel_names for part in name.split("/")
    )
