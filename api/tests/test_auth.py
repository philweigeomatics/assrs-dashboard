"""
Token verification: every way a token can be wrong must be rejected.

Offline. Generates a throwaway P-256 key, publishes it as a JWKS file, and
points the verifier at it with a file:// URL, so the real verification code
runs end to end without Supabase.

    python -m pytest api/tests/test_auth.py -q
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

ISS = "https://example.supabase.co/auth/v1"


def _jwk_for(private_key, kid):
    pub = jwt.algorithms.ECAlgorithm.to_jwk(private_key.public_key(), as_dict=True)
    pub.update({"kid": kid, "alg": "ES256", "use": "sig"})
    return pub


@pytest.fixture()
def keys(tmp_path, monkeypatch):
    good = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    kid = str(uuid.uuid4())
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [_jwk_for(good, kid)]}))

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co/rest/v1/")
    monkeypatch.setenv("SUPABASE_JWKS_URL", jwks.as_uri())
    import api.auth as auth
    auth._jwk_client = None                     # fresh client per test
    return auth, good, other, kid


def _token(key, kid, **over):
    now = int(time.time())
    claims = {"sub": str(uuid.uuid4()), "aud": "authenticated", "role": "authenticated",
              "iss": ISS, "email": "me@example.com", "iat": now, "exp": now + 3600,
              "is_anonymous": False}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm="ES256", headers={"kid": kid})


def test_valid_token_accepted(keys):
    auth, good, _, kid = keys
    claims = auth.verify_token(_token(good, kid))
    assert claims["email"] == "me@example.com"


def test_rest_suffix_in_url_is_tolerated(keys):
    auth, *_ = keys
    assert auth._issuer() == ISS


@pytest.mark.parametrize("label,over", [
    ("expired", {"exp": int(time.time()) - 10}),
    ("wrong audience", {"aud": "anon"}),
    ("wrong issuer", {"iss": "https://evil.supabase.co/auth/v1"}),
    ("anon role", {"role": "anon"}),
    ("anonymous user", {"is_anonymous": True}),
    ("no exp", {"exp": None}),
])
def test_bad_claims_rejected(keys, label, over):
    auth, good, _, kid = keys
    with pytest.raises(auth.HTTPException) as e:
        auth.verify_token(_token(good, kid, **over))
    assert e.value.status_code == 401, label


def test_signed_by_another_key_rejected(keys):
    auth, _, other, kid = keys
    with pytest.raises(auth.HTTPException) as e:
        auth.verify_token(_token(other, kid))          # right kid, wrong key
    assert e.value.status_code == 401


def test_hs256_forgery_rejected(keys):
    """Classic confusion attack: HS256 'signed' with a guessable secret."""
    auth, _, _, kid = keys
    forged = jwt.encode({"sub": "x", "aud": "authenticated", "role": "authenticated",
                         "iss": ISS, "exp": int(time.time()) + 60},
                        "secret", algorithm="HS256", headers={"kid": kid})
    with pytest.raises(auth.HTTPException):
        auth.verify_token(forged)


def test_missing_bearer_is_401(keys):
    from fastapi.testclient import TestClient
    from api.main import app
    r = TestClient(app).get("/me")
    assert r.status_code == 401
    r = TestClient(app).get("/me", headers={"Authorization": "Token abc"})
    assert r.status_code == 401


def test_health_needs_no_token(keys):
    from fastapi.testclient import TestClient
    from api.main import app
    assert TestClient(app).get("/health").json() == {"ok": True}


def test_bad_ticker_rejected_before_any_work(keys, monkeypatch):
    from fastapi.testclient import TestClient
    import api.auth as auth
    from api.main import app
    app.dependency_overrides[auth.current_user] = lambda: auth.AppUser(
        id=1, username="t", email="t@x", role="user", auth_user_id="u")
    try:
        r = TestClient(app).get("/analysis/60051a")
        assert r.status_code == 422
    finally:
        app.dependency_overrides.clear()
