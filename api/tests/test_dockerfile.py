"""
The image must contain everything the API imports.

This guards one specific, expensive mistake. `COPY *.py ./` takes FILES only,
so adding a top-level PACKAGE (a directory with __init__.py) and forgetting its
own COPY line leaves it out of the image. Nothing fails at build time — the
build is green, the push is green — and then uvicorn cannot import the app, no
process ever binds to $PORT, and Cloud Run reports:

    The user-provided container failed to start and listen on the port
    defined provided by the PORT=8080 environment variable

which says nothing about an import and takes several minutes to say it. That is
exactly what adding the `markets` package did.

    python -m pytest api/tests/test_dockerfile.py -q
"""

from __future__ import annotations

import ast
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

DOCKERFILE = os.path.join(ROOT, "Dockerfile")

#: Directories that are Python packages but have no business in the API image.
NOT_SHIPPED = {"pages", "supabase", ".claude", "web"}


def _dockerfile() -> str:
    with open(DOCKERFILE, encoding="utf-8") as fh:
        return fh.read()


def _copied_paths() -> set[str]:
    """Source paths named by a COPY line, minus the destination argument."""
    out: set[str] = set()
    for line in _dockerfile().splitlines():
        line = line.strip()
        if not line.upper().startswith("COPY "):
            continue
        parts = re.split(r"\s+", line)[1:]
        parts = [p for p in parts if not p.startswith("--")]
        for src in parts[:-1]:                     # last argument is the dest
            out.add(src.rstrip("/"))
    return out


def _local_packages() -> set[str]:
    """Top-level importable packages in the repo (directories with __init__.py)."""
    out = set()
    for name in os.listdir(ROOT):
        if name.startswith(".") or name in NOT_SHIPPED:
            continue
        if os.path.isfile(os.path.join(ROOT, name, "__init__.py")):
            out.add(name)
    return out


def _imports_of(path: str) -> set[str]:
    """Top-level module names imported by one file."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_local_package_the_api_imports_is_copied():
    """The regression this file exists for."""
    copied = _copied_paths()
    packages = _local_packages()
    imported = _imports_of(os.path.join(ROOT, "api", "main.py"))

    missing = sorted(p for p in packages & imported if p not in copied)
    assert not missing, (
        f"api/main.py imports {missing}, which the Dockerfile never COPYs. "
        f"The build will succeed and the container will fail to start. "
        f"Add `COPY {missing[0]} ./{missing[0]}`.")


def test_every_local_package_is_copied_even_if_imported_lazily():
    """
    Wider net than the test above: a package reached only through a runtime
    import — markets.get() pulls in markets.cn on the first A-share request —
    is just as absent from the image, and fails on a request instead of at
    startup, which is harder to notice.
    """
    copied = _copied_paths()
    missing = sorted(p for p in _local_packages() if p not in copied)
    assert not missing, f"packages missing a COPY line in the Dockerfile: {missing}"


def test_the_entrypoint_module_is_reachable():
    assert "api" in _copied_paths()
    assert os.path.isfile(os.path.join(ROOT, "api", "main.py"))


@pytest.mark.parametrize("needed", ["requirements.txt", "api/requirements.txt"])
def test_requirements_are_installed_in_the_image(needed):
    assert needed in _copied_paths()


def test_yfinance_is_declared():
    """
    The North American adapter imports it lazily, so a missing dependency
    surfaces as a 503 on the first US symbol rather than at startup.
    """
    with open(os.path.join(ROOT, "api", "requirements.txt"), encoding="utf-8") as fh:
        assert "yfinance" in fh.read()


def test_the_container_listens_on_cloud_runs_port():
    """Cloud Run injects $PORT and kills anything that binds elsewhere."""
    text = _dockerfile()
    cmd = next(l for l in text.splitlines() if l.strip().startswith("CMD"))
    assert "${PORT}" in cmd or "$PORT" in cmd
    assert "0.0.0.0" in cmd, "binding to localhost is invisible to Cloud Run"
