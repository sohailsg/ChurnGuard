import pytest
from sqlalchemy import text

from tools import validate_sql

A = "A-10000"


def ok(q, a=A):
    return validate_sql(q, a, "sqlite")


def bad(q, a=A):
    with pytest.raises(ValueError):
        validate_sql(q, a, "sqlite")


def test_accepts_usage_query():
    ok(f"SELECT feature, SUM(sessions) FROM usage_daily WHERE account_id = '{A}' GROUP BY feature")


def test_accepts_cte():
    ok(f"WITH u AS (SELECT * FROM usage_daily WHERE account_id = '{A}') SELECT COUNT(*) FROM u")


def test_accepts_count_star_on_accounts():
    ok(f"SELECT COUNT(*) FROM accounts WHERE account_id = '{A}'")


@pytest.mark.parametrize("q", [
    f"DELETE FROM usage_daily WHERE account_id = '{A}'",
    f"DROP TABLE accounts -- {A}",
    f"UPDATE accounts SET arr = 0 WHERE account_id = '{A}'",
    f"SELECT 1 FROM usage_daily WHERE account_id = '{A}'; DROP TABLE accounts",
    f"SELECT * FROM playbook WHERE account_id = '{A}'",
    f"SELECT contact_email FROM accounts WHERE account_id = '{A}'",
    f"SELECT contact_name FROM accounts WHERE account_id = '{A}'",
    f"SELECT * FROM accounts WHERE account_id = '{A}'",
    "SELECT SUM(sessions) FROM usage_daily",
    "SELECT SUM(sessions) FROM usage_daily WHERE account_id = 'A-10001'",
    f"SELECT SUM(sessions) FROM usage_daily WHERE account_id = '{A}' OR account_id = 'A-10001'",
    "not sql at all (((",
])
def test_rejects(q):
    bad(q)


def test_agent_engine_is_read_only(seeded):
    from core import agent_engine
    with agent_engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM accounts")).scalar() > 0
    with pytest.raises(Exception):
        with agent_engine.begin() as c:
            c.execute(text("UPDATE accounts SET arr = 0"))
