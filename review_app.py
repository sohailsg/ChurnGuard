"""CSM review queue (approve / edit / reject / mark sent). Run: python review_app.py  -> http://127.0.0.1:5000"""
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import quote

import pandas as pd
from flask import Flask, flash, redirect, render_template_string, request, url_for
from sqlalchemy import insert, text, update

from core import engine, guardrail_rules, interventions
from guardrails import draft_issues, permits_for

REJECT_REASONS = ["Wrong root cause", "Account not actually at risk", "Tone or wording", "Unapproved promise",
                  "Already in contact with customer", "Other"]
STATUSES = ["pending_review", "approved", "edited", "sent", "rejected", "guardrail_failed", "error", "holdout"]

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "churnguard-local")  # override for any shared deployment

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChurnGuard review</title>
<style>
 :root{--bg:#f6f7f9;--card:#fff;--ink:#1c2430;--mute:#6b7685;--line:#dfe3e8;--acc:#2f5fd0;--ok:#1f8a4c;--bad:#c0392b}
 @media(prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e6e9ee;--mute:#98a2b0;--line:#2d333c;--acc:#6f95ee;--ok:#4cc17f;--bad:#f0705f}}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
 header{padding:16px 24px;border-bottom:1px solid var(--line);background:var(--card);display:flex;gap:16px;flex-wrap:wrap;align-items:center}
 h1{font-size:18px;margin:0 16px 0 0} main{max-width:1200px;margin:0 auto;padding:16px 24px}
 nav a{display:inline-block;padding:4px 10px;margin:2px;border:1px solid var(--line);border-radius:6px;color:var(--ink);text-decoration:none;font-size:13px}
 nav a.on{background:var(--acc);border-color:var(--acc);color:#fff} nav b{color:var(--mute);font-weight:500;margin-left:4px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
 .card h2{font-size:16px;margin:0 0 4px} .meta{color:var(--mute);font-size:13px;margin-bottom:12px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:20px} @media(max-width:800px){.cols{grid-template-columns:1fr}}
 table{border-collapse:collapse;width:100%;font-size:13px} td,th{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
 input,textarea,select{width:100%;padding:7px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit}
 textarea{min-height:240px} label{font-size:12px;color:var(--mute);display:block;margin:8px 0 2px}
 button,.btn{padding:7px 14px;border:0;border-radius:6px;background:var(--acc);color:#fff;font:inherit;cursor:pointer;text-decoration:none;display:inline-block}
 button.rej{background:var(--bad)} .row{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap} .row select{width:auto}
 .err{background:color-mix(in srgb,var(--bad) 14%,transparent);border:1px solid var(--bad);padding:8px 10px;border-radius:6px;margin:8px 0}
 .flash{background:color-mix(in srgb,var(--ok) 14%,transparent);border:1px solid var(--ok);padding:8px 10px;border-radius:6px;margin:8px 0}
 .flash.bad{background:color-mix(in srgb,var(--bad) 14%,transparent);border-color:var(--bad)}
 details summary{cursor:pointer;color:var(--mute);font-size:13px} pre{white-space:pre-wrap;background:var(--bg);padding:10px;border-radius:6px}
</style></head><body>
<header><h1>ChurnGuard review queue</h1>
 <form method="get" style="display:flex;gap:6px;align-items:center;margin:0">
  <input type="hidden" name="status" value="{{ status }}">
  <input name="reviewer" value="{{ reviewer }}" placeholder="Reviewer name" style="width:180px"><button>Set</button></form>
 <nav>{% for s in statuses %}<a href="{{ url_for('index', status=s, reviewer=reviewer) }}" class="{{ 'on' if s == status }}">{{ s }}<b>{{ counts.get(s, 0) }}</b></a>{% endfor %}</nav>
</header>
<main>
{% for cat, msg in get_flashed_messages(with_categories=true) %}<div class="flash {{ cat }}">{{ msg }}</div>{% endfor %}
{% if not rows %}<p>Nothing in this queue.</p>{% endif %}
{% for r in rows %}
<div class="card">
 <h2>{{ r.company }} · {{ r.account_id }}</h2>
 <div class="meta">risk {{ '%.2f'|format(r.risk_score) }} ({{ r.risk_tier }}) · ARR ${{ '{:,.0f}'.format(r.arr) }} · {{ r.run_date }}</div>
 {% if not r.prof %}<p>{{ r.error or 'Holdout: control group, no outreach.' }}</p>
 {% else %}
 <div class="cols">
  <section>
   <h3>Evidence</h3>
   <p><b>Likely root cause:</b> {{ r.prof.likely_root_cause }}<br><b>Sentiment:</b> {{ r.prof.sentiment }} · <b>Confidence:</b> {{ r.prof.confidence }}</p>
   <table><tr><th>Signal</th><th>Detail</th><th>Evidence</th></tr>
   {% for s in r.prof.signals %}<tr><td>{{ s.get('name') or s.get('type') or s.get('signal') or '' }}</td><td>{{ s.detail }}</td><td>{{ s.evidence|join(', ') }}</td></tr>{% endfor %}</table>
   <p><b>Strategy:</b> {{ r.iv.strategy }}<br>{{ r.iv.get('rationale', '') }}<br><b>Playbook:</b> {{ r.iv.playbook_refs|join(', ') }}</p>
   {% if r.guardrail_issues %}<div class="err">{{ r.guardrail_issues }}</div>{% endif %}
  </section>
  <section>
   <h3>Draft</h3>
   {% if r.status in ('pending_review', 'guardrail_failed') %}
   <form method="post" action="{{ url_for('act', row_id=r.id) }}">
    <input type="hidden" name="status" value="{{ status }}"><input type="hidden" name="reviewer" value="{{ reviewer }}">
    <label>Subject</label><input name="subject" value="{{ r.final_subject or '' }}">
    <label>Body</label><textarea name="body">{{ r.final_body or '' }}</textarea>
    <div class="row">
     <button name="action" value="approve" {{ '' if reviewer else 'disabled' }}>Approve</button>
     <select name="reason">{% for x in reasons %}<option>{{ x }}</option>{% endfor %}</select>
     <button class="rej" name="action" value="reject" {{ '' if reviewer else 'disabled' }}>Reject</button>
    </div>
    {% if not reviewer %}<p class="meta">Enter your name at the top to approve or reject.</p>{% endif %}
   </form>
   {% elif r.status in ('approved', 'edited') %}
   <label>Ready to send</label><pre>{{ r.final_subject }}

{{ r.filled }}</pre>
   <div class="row"><a class="btn" href="mailto:{{ r.contact_email }}?subject={{ r.q_subject }}&body={{ r.q_body }}">Open in mail client</a>
   <form method="post" action="{{ url_for('act', row_id=r.id) }}" style="margin:0">
    <input type="hidden" name="status" value="{{ status }}"><input type="hidden" name="reviewer" value="{{ reviewer }}">
    <button name="action" value="sent">Mark as sent</button></form></div>
   {% else %}<pre>{{ r.final_subject }}

{{ r.final_body }}</pre>{% if r.rejection_reason %}<p class="meta">Rejected: {{ r.rejection_reason }}</p>{% endif %}{% endif %}
  </section>
 </div>
 {% endif %}
</div>
{% endfor %}
<details class="card"><summary>Add guardrail rule</summary>
 <form method="post" action="{{ url_for('add_rule') }}">
  <input type="hidden" name="status" value="{{ status }}"><input type="hidden" name="reviewer" value="{{ reviewer }}">
  <label>Regex (case-insensitive)</label><input name="pattern"><label>Reason</label><input name="reason">
  <div class="row"><button>Add rule</button></div></form></details>
</main></body></html>"""


def _back(**extra):
    return redirect(url_for("index", status=request.form.get("status", "pending_review"),
                            reviewer=request.form.get("reviewer", ""), **extra))


@app.get("/")
def index():
    status = request.args.get("status", "pending_review")
    reviewer = request.args.get("reviewer", "")
    counts = dict(pd.read_sql(text("SELECT status, COUNT(*) AS n FROM churn_interventions GROUP BY status"), engine).values)
    q = pd.read_sql(text("SELECT i.*, a.name AS company, a.csm_owner, a.contact_name, a.contact_email "
                         "FROM churn_interventions i JOIN accounts a ON a.account_id = i.account_id "
                         "WHERE i.status = :s ORDER BY i.risk_score * i.arr DESC"), engine, params={"s": status})
    rows = []
    for r in q.to_dict("records"):
        r["prof"] = json.loads(r["profile_json"]) if r["profile_json"] else None
        r["iv"] = json.loads(r["intervention_json"]) if r["intervention_json"] else None
        first = r["contact_name"].split()[0] if r["contact_name"] else "there"
        r["filled"] = (r["final_body"] or "").replace("[CONTACT_FIRST_NAME]", first).replace("[CSM_NAME]", r["csm_owner"] or "")
        r["q_subject"], r["q_body"] = quote(r["final_subject"] or ""), quote(r["filled"])
        rows.append(r)
    return render_template_string(PAGE, rows=rows, counts=counts, status=status, statuses=STATUSES,
                                  reviewer=reviewer, reasons=REJECT_REASONS)


@app.post("/act/<int:row_id>")
def act(row_id: int):
    action, reviewer = request.form["action"], request.form.get("reviewer", "").strip()

    def save(**vals):
        with engine.begin() as c:
            c.execute(update(interventions).where(interventions.c.id == row_id).values(**vals))

    if action == "sent":
        save(status="sent", sent_at=datetime.now())
        flash("Marked as sent.", "ok")
        return _back()
    if not reviewer:
        flash("Enter your name at the top first.", "bad")
        return _back()

    if action == "reject":
        save(status="rejected", rejection_reason=request.form.get("reason", "Other"),
             reviewer=reviewer, reviewed_at=datetime.now())
        flash("Rejected.", "ok")
        return _back()

    with engine.connect() as c:
        r = c.execute(text("SELECT i.account_id, i.intervention_json, a.contact_name FROM churn_interventions i "
                           "JOIN accounts a ON a.account_id = i.account_id WHERE i.id = :i"), {"i": row_id}).mappings().one()
    iv = json.loads(r["intervention_json"])
    subj, body = request.form["subject"], request.form["body"]
    issues = draft_issues(subj, body, r["account_id"], [r["contact_name"]] if r["contact_name"] else [],
                          permits_for(iv["playbook_refs"]))
    if issues:
        flash("Edits fail guardrails: " + "; ".join(issues), "bad")
        return _back()
    orig = f"{iv['draft_email']['subject']}\n{iv['draft_email']['body']}"
    sim = SequenceMatcher(None, orig, f"{subj}\n{body}").ratio()
    save(status="approved" if sim > 0.999 else "edited", final_subject=subj, final_body=body,
         edit_similarity=sim, reviewer=reviewer, reviewed_at=datetime.now())
    flash("Approved." if sim > 0.999 else "Saved as edited and approved.", "ok")
    return _back()


@app.post("/rules")
def add_rule():
    pattern, reason = request.form.get("pattern", "").strip(), request.form.get("reason", "").strip()
    if not pattern:
        flash("Enter a regex.", "bad")
        return _back()
    try:
        re.compile(pattern)
    except re.error as e:
        flash(f"Invalid regex: {e}", "bad")
        return _back()
    with engine.begin() as c:
        c.execute(insert(guardrail_rules).values(pattern=pattern, reason=reason or pattern, active=True,
                                                 created_at=datetime.now()))
    flash("Rule added.", "ok")
    return _back()


if __name__ == "__main__":
    app.run(debug=False)
