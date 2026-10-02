"""Isolated test environment: temp SQLite DB + temp chroma dir, set BEFORE core is imported (engine is built at import)."""
import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="churnguard-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_tmp, 'test.db').as_posix()}"
os.environ.pop("AGENT_DATABASE_URL", None)
os.environ["CHROMA_PATH"] = str(Path(_tmp, "chroma"))
os.environ["MODEL_PATH"] = str(Path(_tmp, "model.joblib"))
os.environ.pop("ZENDESK_SUBDOMAIN", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import core  # noqa: E402


@pytest.fixture(scope="session")
def seeded():
    core.seed_synthetic(n_accounts=40, days=150, seed=7)
    return core


@pytest.fixture()
def acct(seeded):
    """(account_id, contact_name) of a seeded account."""
    from sqlalchemy import text
    with core.engine.connect() as c:
        r = c.execute(text("SELECT account_id, contact_name, name FROM accounts ORDER BY account_id LIMIT 1")).one()
    return r.account_id, r.contact_name, r.name
