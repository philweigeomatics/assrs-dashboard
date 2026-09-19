"""
No literal route may be shadowed by a parameterised one declared above it.

Starlette matches in declaration order. `/strategies/{name}` declared before
`/strategies/discover` means a GET of the literal path never reaches its own
handler — it reaches the parameterised one and fails that handler's path
pattern, so the browser gets a 422 about a field the caller never sent. It
looks like a client bug and it is a routing bug.

Nothing about this is visible in either function, which is why it is a test
and not a comment.

    python -m pytest api/tests/test_route_order.py -q
"""

from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from api.main import app  # noqa: E402


def routes():
    """(index, path, methods) for every route that has a path."""
    out = []
    for i, r in enumerate(app.routes):
        path = getattr(r, "path", None)
        if path:
            out.append((i, path, set(getattr(r, "methods", ()) or ())))
    return out


def as_regex(template: str) -> re.Pattern:
    """`/a/{b}/c` → a pattern matching `/a/anything/c`."""
    return re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", re.escape(template)
                                   .replace(r"\{", "{").replace(r"\}", "}")) + "$")


def test_no_literal_path_is_shadowed_by_an_earlier_templated_one():
    rs = routes()
    shadowed = []
    for i, path, methods in rs:
        if "{" in path:
            continue
        for j, other, other_methods in rs:
            if j >= i or "{" not in other:
                continue
            if methods & other_methods and as_regex(other).match(path):
                shadowed.append(f"{sorted(methods)} {path} ← {other} (declared earlier)")
    assert not shadowed, "these literal routes are unreachable:\n  " + "\n  ".join(shadowed)


@pytest.mark.parametrize("method, path", [
    ("GET", "/strategies/discover"),
    ("POST", "/strategies/discover"),
    ("POST", "/strategies/pair-trade"),
    ("POST", "/strategies/lead-lag"),
])
def test_the_routes_that_were_actually_broken_resolve_to_themselves(method, path):
    """
    A page load on 配对交易 asked for GET /strategies/discover and got
    '"discover" should match ^(t-trading|mean-reversion)$' — the parameterised
    handler answering for a path that was never meant for it.
    """
    first = next(r for r in app.routes
                 if getattr(r, "path", None)
                 and method in (getattr(r, "methods", ()) or ())
                 and as_regex(r.path).match(path))
    assert first.path == path, (
        f"{method} {path} is answered by {first.path}, not by its own handler")


def test_the_ordering_survives_a_route_added_in_the_wrong_place():
    """
    The guarantee has to hold for routes that do not exist yet, or the next
    `/strategies/<something>` reintroduces the same 422.
    """
    from fastapi import FastAPI

    from api.main import _literal_routes_first

    probe = FastAPI()

    @probe.get("/x/{name}")
    def templated(name: str):                                      # noqa: ANN202
        return name

    @probe.get("/x/literal")
    def literal():                                                 # noqa: ANN202
        return "literal"

    paths = [getattr(r, "path", "") for r in probe.routes]
    assert paths.index("/x/{name}") < paths.index("/x/literal"), "fixture is wrong"

    _literal_routes_first(probe)
    paths = [getattr(r, "path", "") for r in probe.routes]
    assert paths.index("/x/literal") < paths.index("/x/{name}")


def test_reordering_keeps_every_route():
    """A partition that drops a route would be a far worse bug than the one
    it fixes."""
    from fastapi import FastAPI

    from api.main import _literal_routes_first

    probe = FastAPI()
    for p in ("/a", "/b/{x}", "/c", "/d/{y}/z"):
        probe.get(p)(lambda: None)

    before = {(getattr(r, "path", ""), id(r)) for r in probe.routes}
    _literal_routes_first(probe)
    assert {(getattr(r, "path", ""), id(r)) for r in probe.routes} == before


def test_relative_order_within_each_group_is_untouched():
    """
    Stable, because routes whose paths overlap in other ways (two templated
    ones, say) still rely on the order they were declared in.
    """
    from fastapi import FastAPI

    from api.main import _literal_routes_first

    probe = FastAPI()
    for p in ("/t/{a}", "/lit1", "/t/{b}/x", "/lit2"):
        probe.get(p)(lambda: None)

    _literal_routes_first(probe)
    paths = [getattr(r, "path", "") for r in probe.routes]
    assert paths.index("/lit1") < paths.index("/lit2")
    assert paths.index("/t/{a}") < paths.index("/t/{b}/x")
