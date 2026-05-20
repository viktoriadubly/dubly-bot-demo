"""
Dubly Support-Agent — Web-Demo (Streamlit)
==========================================
Eine schoene browser-basierte Demo des Bots, zum Teilen mit Kollegen.
Nutzt Anthropic-SDK direkt (statt claude-agent-sdk), damit's stressfrei
in der Cloud deploybar ist.

Lokal starten:
    source .venv/bin/activate
    streamlit run dubly_agent_web.py

Cloud-Deployment: siehe 16_Web-Demo-Anleitung.docx
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import uuid
import zoneinfo
from pathlib import Path

# ---------------------------------------------------------------------------
# .env laden (mini-Parser)
# ---------------------------------------------------------------------------
def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v

_load_env(Path(__file__).parent / ".env")

try:
    import streamlit as st
    import anthropic
    import numpy as np
    import requests
    from google import genai
    from google.genai import types
except ImportError as e:  # pragma: no cover
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"[FEHLER] Bibliothek fehlt: {missing}")
    print("        Bitte einmal im Terminal:")
    print("            pip install streamlit anthropic google-genai numpy requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Konfig
# ---------------------------------------------------------------------------
def _secret(name: str) -> str:
    """Liest Secret aus st.secrets, faellt still auf '' zurueck wenn keine secrets.toml da ist."""
    try:
        return (st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip() or _secret("ANTHROPIC_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or _secret("GEMINI_API_KEY")
PLAIN_KEY = os.environ.get("PLAIN_API_KEY", "").strip() or _secret("PLAIN_API_KEY")

CHUNKS_FILE = Path(__file__).parent / "chunks_with_embeddings.json"
MOCK_DB_FILE = Path(__file__).parent / "mock_customers.json"
FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"
AUDIT_LOG_FILE = Path(__file__).parent / "audit_log.jsonl"

EMBEDDING_MODEL = "gemini-embedding-001"
CLAUDE_MODEL = "claude-haiku-4-5"
TOP_K_DEFAULT = 5
MIN_SIMILARITY = 0.55
MAX_TURN_STEPS = 12  # Hartes Limit fuer Tool-Loop pro User-Turn

# Plain
PLAIN_API_URL = "https://core-api.uk.plain.com/graphql/v1"
PLAIN_TEST_PREFIX = "[BOT-TEST] "
PLAIN_FEEDBACK_PREFIX = "[BOT-FEEDBACK] "
PLAIN_BOT_TEST_LABEL_ID = os.environ.get("PLAIN_BOT_TEST_LABEL_ID", "").strip()


# ---------------------------------------------------------------------------
# Wissensbasis + Mock-DB laden (gecached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_chunks():
    if not CHUNKS_FILE.exists():
        return None, None
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return chunks, embeddings


@st.cache_resource
def load_mock_db():
    if not MOCK_DB_FILE.exists():
        return None
    with MOCK_DB_FILE.open(encoding="utf-8") as f:
        raw = json.load(f)
    return raw


@st.cache_resource
def get_gemini():
    return genai.Client(api_key=GEMINI_KEY)


@st.cache_resource
def get_anthropic():
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _err(msg: str) -> str:
    return f"ERROR: {msg}"


def _ok(payload) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    return st.session_state.session_id


def _audit(tool_name: str, args: dict, status: str, detail: dict | None = None) -> None:
    entry = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "session_id": _session_id(),
        "source": "web",
        "tool": tool_name,
        "args": args,
        "status": status,
        "detail": detail or {},
    }
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _persist_mock_db(raw: dict) -> None:
    MOCK_DB_FILE.write_text(json.dumps(raw, indent=2, ensure_ascii=False))


def _customer_summary(c: dict) -> dict:
    return {
        "id": c["id"],
        "email": c["email"],
        "name": c["name"],
        "joined_at": c.get("joined_at"),
        "plan": c.get("plan"),
    }


# ---------------------------------------------------------------------------
# Retrieval (KB)
# ---------------------------------------------------------------------------
def _embed_query(text: str) -> np.ndarray:
    res = get_gemini().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    e = np.array(res.embeddings[0].values, dtype=np.float32)
    return e / np.linalg.norm(e)


def _retrieve(query: str, top_k: int) -> list[dict]:
    chunks, embeddings = load_chunks()
    if chunks is None:
        return []
    q_emb = _embed_query(query)
    sims = embeddings @ q_emb
    idx = np.argsort(sims)[-top_k:][::-1]
    out = []
    for i in idx:
        c = chunks[int(i)]
        out.append({
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "score": round(float(sims[i]), 3),
            "text": c.get("text", "")[:1200],
        })
    return out


# ---------------------------------------------------------------------------
# Plain
# ---------------------------------------------------------------------------
def _plain_request(query: str, variables: dict) -> dict:
    if not PLAIN_KEY:
        return {"_no_key": True}
    try:
        resp = requests.post(
            PLAIN_API_URL,
            headers={"Authorization": f"Bearer {PLAIN_KEY}", "Content-Type": "application/json"},
            json={"query": query, "variables": variables},
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        return {"errors": [{"message": f"Verbindungsfehler: {e}"}]}
    if resp.status_code != 200:
        return {"errors": [{"message": f"HTTP {resp.status_code}: {resp.text[:300]}"}]}
    return resp.json()


_PLAIN_UPSERT = """
mutation upsertCustomer($input: UpsertCustomerInput!) {
  upsertCustomer(input: $input) {
    result customer { id } error { message code }
  }
}
"""
_PLAIN_THREAD = """
mutation createThread($input: CreateThreadInput!) {
  createThread(input: $input) {
    thread { id title } error { message code }
  }
}
"""
_PLAIN_NOTE = """
mutation createNote($input: CreateNoteInput!) {
  createNote(input: $input) {
    note { id } error { message code }
  }
}
"""


def _plain_upsert(email: str, name: str) -> tuple[str | None, str | None]:
    p = _plain_request(_PLAIN_UPSERT, {
        "input": {
            "identifier": {"emailAddress": email},
            "onCreate": {"fullName": name, "email": {"email": email, "isVerified": False}},
            "onUpdate": {},
        }
    })
    if p.get("_no_key"): return None, "Plain nicht konfiguriert"
    if p.get("errors"): return None, str(p["errors"])[:300]
    data = (p.get("data") or {}).get("upsertCustomer") or {}
    if data.get("error"): return None, str(data["error"])
    return ((data.get("customer") or {}).get("id"), None)


def _plain_create_thread(customer_id: str, title: str, body: str, prefix: str = PLAIN_TEST_PREFIX) -> tuple[str | None, str | None]:
    full = title if title.startswith(prefix) else prefix + title
    inp: dict = {
        "title": full[:120],
        "customerIdentifier": {"customerId": customer_id},
        "components": [{"componentText": {"text": body[:9000]}}],
    }
    if PLAIN_BOT_TEST_LABEL_ID:
        inp["labelTypeIds"] = [PLAIN_BOT_TEST_LABEL_ID]
    p = _plain_request(_PLAIN_THREAD, {"input": inp})
    if p.get("_no_key"): return None, "Plain nicht konfiguriert"
    if p.get("errors"): return None, str(p["errors"])[:300]
    data = (p.get("data") or {}).get("createThread") or {}
    if data.get("error"): return None, str(data["error"])
    return ((data.get("thread") or {}).get("id"), None)


def _plain_add_note(thread_id: str, body: str) -> tuple[bool, str | None]:
    p = _plain_request(_PLAIN_NOTE, {"input": {"threadId": thread_id, "body": body[:9000]}})
    if p.get("_no_key"): return False, "Plain nicht konfiguriert"
    if p.get("errors"): return False, str(p["errors"])[:300]
    data = (p.get("data") or {}).get("createNote") or {}
    if data.get("error"): return False, str(data["error"])
    return True, None


# ---------------------------------------------------------------------------
# Tool-Implementierungen (eingebettete Logik, von Anthropic-Tool-Loop dispatcht)
# ---------------------------------------------------------------------------
def tool_search_knowledge_base(args: dict) -> str:
    query = (args.get("query") or "").strip()
    top_k = int(args.get("top_k") or TOP_K_DEFAULT)
    if not query:
        return _err("query fehlt.")
    top_k = max(1, min(top_k, 10))
    hits = _retrieve(query, top_k)
    top = hits[0]["score"] if hits else 0.0
    if top < MIN_SIMILARITY:
        body = f"KEIN ARTIKEL UEBER SCHWELLE. Bester Score: {top}. Sei ehrlich, eskaliere.\n\n"
    else:
        body = f"GEFUNDEN: {len(hits)} Treffer (Top {top}).\n\n"
    for i, h in enumerate(hits, start=1):
        body += f"--- Treffer {i} (Score {h['score']}) ---\nTitel: {h['title']}\nQuelle: {h['url']}\n{h['text']}\n\n"
    return body


def _get_customers():
    raw = load_mock_db()
    if raw is None:
        return [], {}, {}, {}
    customers = raw.get("customers", [])
    by_email = {c["email"].lower(): c for c in customers}
    by_id = {c["id"]: c for c in customers}
    job_index = {j["id"]: {"job": j, "customer": c} for c in customers for j in c.get("jobs", [])}
    return customers, by_email, by_id, job_index


def tool_get_customer(args: dict) -> str:
    email = (args.get("email") or "").strip().lower()
    if not email:
        return _err("email fehlt.")
    _, by_email, _, _ = _get_customers()
    c = by_email.get(email)
    if not c:
        return _err(f"Kein Kunde mit Email '{email}'.")
    return _ok(_customer_summary(c))


def tool_get_subscription(args: dict) -> str:
    cid = (args.get("customer_id") or "").strip()
    if not cid:
        return _err("customer_id fehlt.")
    _, _, by_id, _ = _get_customers()
    c = by_id.get(cid)
    if not c:
        return _err(f"Kein Kunde mit ID '{cid}'.")
    return _ok(c.get("subscription"))


def tool_get_credits(args: dict) -> str:
    cid = (args.get("customer_id") or "").strip()
    if not cid:
        return _err("customer_id fehlt.")
    _, _, by_id, _ = _get_customers()
    c = by_id.get(cid)
    if not c:
        return _err(f"Kein Kunde mit ID '{cid}'.")
    return _ok(c.get("credits"))


def tool_list_recent_jobs(args: dict) -> str:
    cid = (args.get("customer_id") or "").strip()
    limit = int(args.get("limit") or 5)
    if not cid:
        return _err("customer_id fehlt.")
    _, _, by_id, _ = _get_customers()
    c = by_id.get(cid)
    if not c:
        return _err(f"Kein Kunde mit ID '{cid}'.")
    jobs = sorted(c.get("jobs", []), key=lambda j: j.get("created_at", ""), reverse=True)
    return _ok(jobs[: max(1, min(limit, 20))])


def tool_get_job_status(args: dict) -> str:
    jid = (args.get("job_id") or "").strip()
    if not jid:
        return _err("job_id fehlt.")
    _, _, _, job_index = _get_customers()
    e = job_index.get(jid)
    if not e:
        return _err(f"Kein Job mit ID '{jid}'.")
    return _ok(e["job"])


def tool_create_plain_thread(args: dict) -> str:
    if not PLAIN_KEY:
        return _err("Plain nicht konfiguriert.")
    email = (args.get("customer_email") or "").strip().lower()
    name = (args.get("customer_name") or "Dubly User").strip()
    title = (args.get("title") or "Bot-Konversation").strip()
    summary = (args.get("summary") or "").strip()
    if not email:
        return _err("customer_email fehlt.")
    cust_id, err = _plain_upsert(email, name)
    if err:
        return _err(f"Upsert: {err}")
    thread_id, err = _plain_create_thread(cust_id, title, summary)
    if err:
        return _err(f"Thread: {err}")
    return _ok({"thread_id": thread_id, "customer_id": cust_id})


def tool_add_plain_note(args: dict) -> str:
    if not PLAIN_KEY:
        return _err("Plain nicht konfiguriert.")
    tid = (args.get("thread_id") or "").strip()
    body = (args.get("body") or "").strip()
    if not tid or not body:
        return _err("thread_id und body Pflicht.")
    ok, err = _plain_add_note(tid, body)
    if not ok:
        return _err(f"Note: {err}")
    return _ok({"status": "note_added", "thread_id": tid})


def tool_escalate_to_human(args: dict) -> str:
    if not PLAIN_KEY:
        return _err("Plain nicht konfiguriert.")
    email = (args.get("customer_email") or "").strip().lower()
    name = (args.get("customer_name") or "Dubly User").strip()
    reason = (args.get("reason") or "Bot-Eskalation").strip()
    summary = (args.get("summary") or "").strip()
    if not email or not summary:
        return _err("customer_email und summary Pflicht.")
    cust_id, err = _plain_upsert(email, name)
    if err:
        return _err(f"Upsert: {err}")
    title = f"Escalation: {reason}"
    body = (
        f"BOT-ESKALATION (Web-Demo, Stufe C)\n\n"
        f"Kunde: {name} <{email}>\nGrund: {reason}\n\n"
        f"Zusammenfassung vom Bot:\n{summary}\n"
    )
    thread_id, err = _plain_create_thread(cust_id, title, body)
    if err:
        return _err(f"Thread: {err}")
    _plain_add_note(thread_id, f"Bot-Eskalations-Tag: ESCALATE\nGrund: {reason}\nCustomer: {name} <{email}>\n\n{summary}")
    return _ok({"status": "escalated", "thread_id": thread_id, "customer_id": cust_id})


def tool_grant_test_credits(args: dict) -> str:
    cid = (args.get("customer_id") or "").strip()
    try:
        credits = int(args.get("credits") or 0)
    except (TypeError, ValueError):
        _audit("grant_test_credits", args, "failed", {"reason": "invalid credits"})
        return _err("credits muss eine Zahl sein.")
    reason = (args.get("reason") or "").strip()
    if not cid or not reason:
        _audit("grant_test_credits", args, "failed", {"reason": "missing args"})
        return _err("customer_id und reason Pflicht.")
    if not 1 <= credits <= 5:
        _audit("grant_test_credits", args, "failed", {"reason": "credits out of range"})
        return _err("credits muss zwischen 1 und 5 liegen.")
    raw = load_mock_db()
    customers = raw.get("customers", [])
    c = next((x for x in customers if x["id"] == cid), None)
    if not c:
        _audit("grant_test_credits", args, "failed", {"reason": "unknown customer"})
        return _err(f"Kein Kunde mit ID '{cid}'.")
    if c.get("plan") != "free":
        _audit("grant_test_credits", args, "failed", {"reason": "not free", "current_plan": c.get("plan")})
        return _err(f"Kunde ist auf Plan '{c.get('plan')}' — grant_test_credits nur fuer Free-User.")
    cr = c.get("credits") or {}
    old = int(cr.get("remaining", 0))
    new = old + credits
    cr["remaining"] = new
    _persist_mock_db(raw)
    load_mock_db.clear()  # invalidate cache
    _audit("grant_test_credits", args, "executed",
           {"customer_id": cid, "old_remaining": old, "new_remaining": new, "credits_added": credits, "reason": reason})
    return _ok({"status": "test_credits_granted", "customer_id": cid, "old_remaining": old, "new_remaining": new, "credits_added": credits})


def tool_restart_lipsync_job(args: dict) -> str:
    jid = (args.get("job_id") or "").strip()
    if not jid:
        _audit("restart_lipsync_job", args, "failed", {"reason": "no job_id"})
        return _err("job_id fehlt.")
    raw = load_mock_db()
    for c in raw.get("customers", []):
        for job in c.get("jobs", []):
            if job["id"] == jid:
                if job.get("status") != "failed":
                    _audit("restart_lipsync_job", args, "failed",
                           {"reason": "not failed", "current_status": job.get("status")})
                    return _err(f"Job {jid} hat Status '{job.get('status')}', nur 'failed' restartbar.")
                old_status = job["status"]
                job["status"] = "queued"
                job.pop("error", None)
                job.pop("error_message", None)
                job["restarted_at"] = dt.datetime.utcnow().isoformat() + "Z"
                _persist_mock_db(raw)
                load_mock_db.clear()
                _audit("restart_lipsync_job", args, "executed",
                       {"job_id": jid, "old_status": old_status, "new_status": "queued"})
                return _ok({"status": "restarted", "job_id": jid, "old_status": old_status, "new_status": "queued"})
    _audit("restart_lipsync_job", args, "failed", {"reason": "unknown job"})
    return _err(f"Kein Job mit ID '{jid}'.")


def tool_apply_credit_bonus(args: dict) -> str:
    cid = (args.get("customer_id") or "").strip()
    try:
        credits = int(args.get("credits") or 0)
    except (TypeError, ValueError):
        _audit("apply_credit_bonus", args, "failed", {"reason": "invalid credits"})
        return _err("credits muss eine Zahl sein.")
    reason = (args.get("reason") or "").strip()
    if not cid or not reason:
        _audit("apply_credit_bonus", args, "failed", {"reason": "missing args"})
        return _err("customer_id und reason Pflicht.")
    if not 1 <= credits <= 20:
        _audit("apply_credit_bonus", args, "failed", {"reason": "credits out of range"})
        return _err("credits muss zwischen 1 und 20 liegen.")
    raw = load_mock_db()
    c = next((x for x in raw.get("customers", []) if x["id"] == cid), None)
    if not c:
        _audit("apply_credit_bonus", args, "failed", {"reason": "unknown customer"})
        return _err(f"Kein Kunde mit ID '{cid}'.")
    cr = c.get("credits") or {}
    old = int(cr.get("remaining", 0))
    new = old + credits
    cr["remaining"] = new
    _persist_mock_db(raw)
    load_mock_db.clear()
    _audit("apply_credit_bonus", args, "executed",
           {"customer_id": cid, "old_remaining": old, "new_remaining": new, "credits_added": credits, "reason": reason})
    return _ok({"status": "credits_applied", "customer_id": cid, "old_remaining": old, "new_remaining": new, "credits_added": credits})


# Map tool name -> handler
TOOL_HANDLERS = {
    "search_knowledge_base": tool_search_knowledge_base,
    "get_customer": tool_get_customer,
    "get_subscription": tool_get_subscription,
    "get_credits": tool_get_credits,
    "list_recent_jobs": tool_list_recent_jobs,
    "get_job_status": tool_get_job_status,
    "create_plain_thread": tool_create_plain_thread,
    "add_plain_note": tool_add_plain_note,
    "escalate_to_human": tool_escalate_to_human,
    "grant_test_credits": tool_grant_test_credits,
    "restart_lipsync_job": tool_restart_lipsync_job,
    "apply_credit_bonus": tool_apply_credit_bonus,
}

# Anthropic-Tool-Definitionen (JSON-Schema)
ANTHROPIC_TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Durchsucht das Dubly Help Center semantisch. Nutze fuer jede Sachfrage "
            "(Pricing, Features, How-Tos, Refund-Bedingungen, Sprachen). Bei Score < 0.55: "
            "ehrlich 'kenne ich nicht' und eskalieren."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Suchanfrage in der gleichen Sprache wie der User"},
                "top_k": {"type": "integer", "default": 5, "description": "Anzahl Treffer (1-10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_customer",
        "description": "Kunde per Email finden. Liefert customer_id fuer weitere Tools.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "name": "get_subscription",
        "description": "Abo des Kunden: plan, status, renewal, is_paid.",
        "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    },
    {
        "name": "get_credits",
        "description": "Credit-Stand: remaining, monthly_total, last_topup_at.",
        "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    },
    {
        "name": "list_recent_jobs",
        "description": "Letzte Jobs des Kunden. Nutze wenn User von 'meinem Job' spricht ohne ID.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_job_status",
        "description": "Voll-Detail eines Jobs inkl. error_message bei failed Jobs.",
        "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
    },
    {
        "name": "create_plain_thread",
        "description": "Plain-Thread fuer NICHT-DRINGENDE Faelle (Rueckruf, Feature-Anfrage).",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_email": {"type": "string"},
                "customer_name": {"type": "string"},
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["customer_email", "title", "summary"],
        },
    },
    {
        "name": "add_plain_note",
        "description": "Zusatz-Note an bestehenden Thread.",
        "input_schema": {
            "type": "object",
            "properties": {"thread_id": {"type": "string"}, "body": {"type": "string"}},
            "required": ["thread_id", "body"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "DRINGENDE Eskalationen (Refund/Cancel/Beschwerde, User-Wunsch nach Mensch, "
            "emotional aufgeladen). Erstellt Thread + Eskalations-Note in einem Schritt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_email": {"type": "string"},
                "customer_name": {"type": "string"},
                "reason": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["customer_email", "reason", "summary"],
        },
    },
    {
        "name": "grant_test_credits",
        "description": (
            "Test-Credits fuer Free-User in AUSNAHMEFAELLEN (max 5). NUR bei Bug/"
            "Plattform-Fehler unsererseits. NICHT auf 'will mehr testen' geben — "
            "auf Abo-Optionen aus Help Center verweisen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "credits": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["customer_id", "credits", "reason"],
        },
    },
    {
        "name": "restart_lipsync_job",
        "description": "Restart eines failed Jobs (nur bei vermutetem temporaerem Fehler, NICHT bei Input-Problemen).",
        "input_schema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
    },
    {
        "name": "apply_credit_bonus",
        "description": "Kulanz-Credits fuer PAID-User (max 20) bei Bot-/Plattform-Fehlern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "credits": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["customer_id", "credits", "reason"],
        },
    },
]


# ---------------------------------------------------------------------------
# SLA-Logik
# ---------------------------------------------------------------------------
DUBLY_TZ = zoneinfo.ZoneInfo("Europe/Berlin")
WORKDAY_END_HOUR = 17


def build_sla_context() -> str:
    now = dt.datetime.now(DUBLY_TZ)
    weekday = now.weekday()
    if weekday >= 5:
        situation = "Wochenende"
        normal_reply = ("'We'll come back to you next business day morning.' / "
                        "'Wir melden uns am naechsten Werktag frueh.'")
    elif now.hour >= WORKDAY_END_HOUR:
        situation = "Werktag nach 17 Uhr"
        normal_reply = ("'We'll come back to you tomorrow morning.' / "
                        "'Wir melden uns morgen frueh bei dir.'")
    else:
        situation = "Werktag (Mo-Fr) vor 17 Uhr"
        normal_reply = ("'Our team will come back to you with an update today.' / "
                        "'Wir melden uns heute noch bei dir.'")
    return (
        f"# AKTUELLE ZEIT-INFO\n"
        f"- Jetzt: {now.strftime('%A, %d.%m.%Y %H:%M')} (Europe/Berlin)\n"
        f"- Situation: {situation}\n"
        f"- Anfrage -> {normal_reply}"
    )


# ---------------------------------------------------------------------------
# System-Prompt (gekürzt für Web — exakt das gleiche Verhalten wie Terminal)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BASE = """Du bist der KI-Support-Assistent von Dubly.AI -- einer SaaS-Plattform fuer KI-Video-Dubbing. Du sprichst direkt mit Kunden im Chat.

# SPRACH-REGEL (WICHTIGSTE REGEL, NIEMALS BRECHEN)
Antworte IMMER in der Sprache der LETZTEN User-Nachricht. NICHT relevant: Email-Domain, Name des Kunden, was du frueher geantwortet hast. NUR die aktuelle Nachricht zaehlt.

# DEIN CHARAKTER
- Geduzt im Deutschen, "you" im Englischen. Niemals "Sie".
- Warm und kompetent, aber nicht kumpelhaft.
- Kurz im Chat: 1-3 Saetze ODER praezise Rueckfrage.
- Emojis sparsam: max. 1 pro Antwort.

# TERMINOLOGIE (WICHTIG)
Die Tools sprechen intern von "Jobs". Im Chat NIEMALS "Job" sagen.
Stattdessen, je nach Kontext: "dein Video" / "deine Uebersetzung" /
"dein Dub" (bei type=dub) / "dein Lip-Sync" (bei type=lipsync) /
"deine Stimm-Klone" (bei type=voice_clone).
Englisch analog: "your video" / "your translation" / "your dub" /
"your lip sync" / "your voice clone". Niemals "your job".

# QUELLE FUER GESCHAEFTS-FAKTEN
Konkrete Geschaefts-Fakten (Pricing, Plan-Inhalt, Test-Credit-Menge, Refund-Bedingungen, unterstuetzte Sprachen) NIEMALS aus dem Gedaechtnis -- IMMER ueber search_knowledge_base. Wenn Help Center nichts liefert (Score < 0.55): ehrlich "weiss ich nicht" und eskalieren.

# IDENTITY-CHECK
Bevor du Account-Daten preisgibst (Credits, Subscription, Jobs), muss der User seine Email genannt haben. Wenn nicht: hoeflich fragen.

# ESKALATIONS-TRIGGER (escalate_to_human)
- refund, cancel, scammed, kuendigen, Beschwerde, Anwalt
- Geldbetraege ueber 50 EUR
- Emotional aufgeladene User
- Kunde fragt nach Mensch
- Bei zwei Versuchen nicht klar verstanden
- Account-Aktionen (loeschen, Plan aendern)
- Spezifische Bug-Reports die Account-Zugriff brauchen

# ACTION-TOOL-REGELN
1. Action-Tools (grant_test_credits, restart_lipsync_job, apply_credit_bonus) laufen im DEMO-Modus AUTO-APPROVED -- die Web-UI zeigt dem Tester einen "DEMO MODE"-Hinweis. Du musst nicht extra fragen.
2. Vor dem Tool-Call: ein Satz Ankuendigung (Indikativ, KEIN "darf ich?"). Beispiel: "Ich gebe dir 3 Test-Credits, weil dein erster Test durch unseren Worker-Stau verbraucht wurde."
3. Nach Tool-Result: knapp bestaetigen. KEINE Filler-Ausrufe wie "Fertig!", "Perfekt!", "Done!", "Super!".

# GATEKEEPER-PRINZIP
DU entscheidest ob Aktionen gerechtfertigt sind, nicht der User. Bei reiner Bitte "kann ich mehr Test-Credits?" sagst du freundlich NEIN und verweist auf Abo. Nur bei nachvollziehbarem PLATTFORM-Fehler gibst du Kulanz.

# QUELLEN
Wenn du search_knowledge_base genutzt hast, nenne in deiner Antwort die genutzte Quell-URL.

# SLA siehe unten.
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_BASE + "\n\n" + build_sla_context()


# ---------------------------------------------------------------------------
# Chat-Logik: Tool-Loop mit Anthropic-SDK
# ---------------------------------------------------------------------------
def run_turn(conversation: list[dict], user_msg: str) -> tuple[str, list[dict]]:
    """Fuehrt einen User-Turn aus: Anthropic-Call -> ggf. Tool-Loop -> finaler Text.
    Gibt (final_text, tool_trail) zurueck. tool_trail ist eine Liste von
    {tool, args, result_preview}-Dicts zur Anzeige in der UI."""
    client = get_anthropic()
    messages = conversation + [{"role": "user", "content": user_msg}]
    tool_trail: list[dict] = []

    for step in range(MAX_TURN_STEPS):
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=build_system_prompt(),
            tools=ANTHROPIC_TOOLS,
            messages=messages,
        )
        # Antwort entweder Text oder Tool-Use(s)
        assistant_blocks = list(resp.content)
        tool_uses = [b for b in assistant_blocks if b.type == "tool_use"]
        if not tool_uses:
            # Fertig — finaler Text
            text = "".join(b.text for b in assistant_blocks if b.type == "text").strip()
            messages.append({"role": "assistant", "content": assistant_blocks})
            return text, tool_trail
        # Tool-Loop: jeden tool_use ausfuehren
        messages.append({"role": "assistant", "content": assistant_blocks})
        results = []
        for tu in tool_uses:
            name = tu.name
            args = tu.input or {}
            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                result_str = _err(f"Unbekanntes Tool: {name}")
            else:
                try:
                    result_str = handler(args)
                except Exception as e:  # noqa: BLE001
                    result_str = _err(f"Tool-Exception: {type(e).__name__}: {e}")
            tool_trail.append({
                "tool": name,
                "args": args,
                "result_preview": result_str[:400] + ("…" if len(result_str) > 400 else ""),
            })
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_str})
        messages.append({"role": "user", "content": results})

    # Loop-Limit erreicht
    return "Ich brauche zu lange — lass mich dich direkt an einen Menschen weiterleiten.", tool_trail


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
def save_feedback(message_id: str, rating: str, comment: str, transcript: list[dict]) -> None:
    """Schreibt Feedback in feedback.jsonl und optional als Plain-Thread."""
    entry = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "session_id": _session_id(),
        "message_id": message_id,
        "rating": rating,           # "up" | "down" | "bug"
        "comment": comment.strip(),
        "transcript_tail": transcript[-6:],  # letzte 3 Q&A-Paare als Kontext
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Optional: Plain-Thread fuer Feedback
    if PLAIN_KEY and rating in {"down", "bug"}:
        title = f"{rating.upper()}: Feedback aus Web-Demo"
        body_lines = [
            f"Feedback-Typ: {rating}",
            f"Session: {entry['session_id']}",
            f"Zeit: {entry['ts']}",
            f"Kommentar: {comment or '(kein Kommentar)'}",
            "",
            "Letzter Konversations-Ausschnitt:",
        ]
        for m in entry["transcript_tail"]:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "(tool use / structured)"
            body_lines.append(f"\n[{role}]\n{str(content)[:800]}")
        # generischer "Feedback"-Customer
        fb_email = "feedback@dubly-bot-demo.internal"
        cust_id, err = _plain_upsert(fb_email, "Bot Demo Feedback")
        if not err and cust_id:
            _plain_create_thread(cust_id, title, "\n".join(body_lines), prefix=PLAIN_FEEDBACK_PREFIX)


# ---------------------------------------------------------------------------
# Streamlit UI — Dubly-Branding (polished)
# ---------------------------------------------------------------------------
DUBLY_BLACK = "#0A0A0F"
DUBLY_INK = "#15161E"
DUBLY_TEXT = "#0E0E14"
DUBLY_MUTED = "#6B7280"
DUBLY_SUBTLE = "#9CA3AF"
DUBLY_ACCENT = "#7C5CFF"
DUBLY_ACCENT_SOFT = "#EFEAFF"
DUBLY_BG = "#F7F7F9"
DUBLY_CARD = "#FFFFFF"
DUBLY_BORDER = "#E8E8EE"
DUBLY_USER_BG = "#0E0E14"

DUBLY_LOGO_URL = (
    "https://app.dubly.ai/_next/image?url=%2Fimages%2Flogo-dubly-full_dark.png"
    "&w=384&q=75&dpl=dpl_C6PgYs7oS5mpyHjdH3T7tmbhH9Ui"
)

st.set_page_config(
    page_title="Dubly Support — Bot Demo",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ---------- Base ---------- */
  html, body, .stApp {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
    color: {DUBLY_TEXT};
    -webkit-font-smoothing: antialiased;
  }}
  .main {{ background: {DUBLY_BG}; }}
  #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
  .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 7rem !important;
    max-width: 820px;
  }}

  /* ---------- Hero ---------- */
  .dubly-hero {{
    position: relative;
    background:
      radial-gradient(80% 120% at 100% 0%, rgba(124,92,255,0.32) 0%, rgba(124,92,255,0) 60%),
      radial-gradient(60% 100% at 0% 100%, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0) 60%),
      linear-gradient(160deg, {DUBLY_BLACK} 0%, #161724 100%);
    color: #fff;
    padding: 32px 36px;
    border-radius: 22px;
    margin-bottom: 28px;
    box-shadow:
      0 12px 40px -8px rgba(10, 10, 15, 0.28),
      0 2px 8px rgba(10, 10, 15, 0.06);
    overflow: hidden;
  }}
  .dubly-hero::after {{
    content: ""; position: absolute; inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
    background-size: 18px 18px;
    pointer-events: none; opacity: 0.5;
  }}
  .dubly-hero > * {{ position: relative; z-index: 1; }}
  .dubly-logo-row {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 22px;
  }}
  .dubly-logo {{ height: 28px; }}
  .demo-badge {{
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    color: #fff;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.7px;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.16);
  }}
  .demo-badge .pulse {{
    width: 7px; height: 7px;
    background: #FFD166; border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.7);
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(255, 209, 102, 0.55); }}
    70% {{ box-shadow: 0 0 0 8px rgba(255, 209, 102, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255, 209, 102, 0); }}
  }}
  .dubly-hero h1 {{
    margin: 0; font-size: 30px; line-height: 1.15;
    font-weight: 700; letter-spacing: -0.02em;
  }}
  .dubly-hero p.sub {{
    margin: 10px 0 0; font-size: 15px;
    color: rgba(255,255,255,0.72); font-weight: 400; line-height: 1.55;
    max-width: 540px;
  }}
  .trust-row {{
    display: flex; flex-wrap: wrap; gap: 14px;
    margin-top: 24px; padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.08);
    font-size: 12px; color: rgba(255,255,255,0.6);
    letter-spacing: 0.2px; font-weight: 500;
  }}
  .trust-row span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .trust-row .dot {{
    width: 4px; height: 4px; background: rgba(255,255,255,0.4);
    border-radius: 50%;
  }}

  /* ---------- Welcome / Starter Prompts ---------- */
  .welcome-eyebrow {{
    color: {DUBLY_MUTED};
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .welcome-title {{
    font-size: 22px; font-weight: 700; letter-spacing: -0.01em;
    margin: 0 0 8px 0; color: {DUBLY_TEXT};
  }}
  .welcome-sub {{
    color: {DUBLY_MUTED}; font-size: 14.5px; line-height: 1.55;
    margin: 0 0 22px 0;
  }}
  .starter-grid > div[data-testid="column"] > div > div > div > button {{
    width: 100% !important;
    text-align: left !important;
    background: {DUBLY_CARD} !important;
    border: 1px solid {DUBLY_BORDER} !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    height: auto !important;
    min-height: 76px !important;
    color: {DUBLY_TEXT} !important;
    font-weight: 500 !important;
    line-height: 1.45 !important;
    transition: all 0.16s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    white-space: pre-line !important;
  }}
  .starter-grid > div[data-testid="column"] > div > div > div > button:hover {{
    border-color: {DUBLY_ACCENT} !important;
    background: {DUBLY_ACCENT_SOFT} !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px -2px rgba(124,92,255,0.16);
  }}

  /* ---------- Chat ---------- */
  div[data-testid="stChatMessage"] {{
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    box-shadow: none !important;
    margin-bottom: 4px !important;
  }}
  div[data-testid="stChatMessage"] > div:nth-child(2) {{
    background: {DUBLY_CARD};
    border: 1px solid {DUBLY_BORDER};
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    line-height: 1.55;
  }}
  /* User-Bubble dunkel + rechts */
  div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {{
    flex-direction: row-reverse;
  }}
  div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {{
    background: {DUBLY_USER_BG};
    color: #fff;
    border-color: {DUBLY_USER_BG};
    margin-left: auto;
  }}
  div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) > div:nth-child(2) p {{
    color: #fff !important;
  }}
  /* Avatare */
  div[data-testid="chatAvatarIcon-user"],
  div[data-testid="chatAvatarIcon-assistant"] {{
    background: {DUBLY_CARD} !important;
    border: 1px solid {DUBLY_BORDER} !important;
    width: 34px !important;
    height: 34px !important;
    border-radius: 10px !important;
    font-size: 16px !important;
  }}

  /* ---------- Tool-Trail ---------- */
  details {{ margin-top: 10px; }}
  details summary {{
    cursor: pointer;
    color: {DUBLY_MUTED};
    font-size: 12.5px;
    padding: 6px 10px;
    border-radius: 8px;
    background: #F2F2F5;
    border: 1px solid {DUBLY_BORDER};
    user-select: none;
    display: inline-block;
    list-style: none;
    transition: all 0.15s ease;
  }}
  details summary:hover {{ color: {DUBLY_TEXT}; background: #EDEDF1; }}
  details summary::-webkit-details-marker {{ display: none; }}
  details summary::before {{
    content: "▸ ";
    display: inline-block; margin-right: 4px;
    transition: transform 0.2s;
  }}
  details[open] summary::before {{ transform: rotate(90deg); }}
  .tool-line {{
    background: #F7F7F9;
    border: 1px solid {DUBLY_BORDER};
    padding: 8px 12px;
    border-radius: 8px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 11.5px;
    margin: 8px 0 4px;
    color: {DUBLY_TEXT};
    line-height: 1.5;
  }}

  /* ---------- Feedback Buttons ---------- */
  .feedback-section {{ margin-top: 10px; }}

  /* ---------- General Buttons ---------- */
  .stButton > button {{
    border-radius: 10px;
    border: 1px solid {DUBLY_BORDER};
    background: {DUBLY_CARD};
    color: {DUBLY_TEXT};
    font-weight: 500;
    transition: all 0.15s ease;
  }}
  .stButton > button:hover:not(:disabled) {{
    border-color: {DUBLY_ACCENT};
    color: {DUBLY_ACCENT};
  }}
  .stButton > button[kind="primary"] {{
    background: {DUBLY_TEXT};
    color: #fff;
    border-color: {DUBLY_TEXT};
  }}
  .stButton > button[kind="primary"]:hover:not(:disabled) {{
    background: {DUBLY_INK};
    border-color: {DUBLY_INK};
    color: #fff;
  }}

  /* ---------- Sidebar ---------- */
  section[data-testid="stSidebar"] {{
    background: {DUBLY_CARD};
    border-right: 1px solid {DUBLY_BORDER};
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
  }}
  section[data-testid="stSidebar"] h3 {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
    color: {DUBLY_SUBTLE}; font-weight: 700; margin: 22px 0 10px;
  }}
  .sb-customer {{
    background: #F7F7F9;
    border: 1px solid {DUBLY_BORDER};
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    line-height: 1.45;
  }}
  .sb-customer .sb-email {{
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 11.5px;
    color: {DUBLY_ACCENT};
    word-break: break-all;
  }}
  .sb-customer .sb-meta {{
    color: {DUBLY_MUTED}; font-size: 12px; margin-top: 2px;
  }}
  .sb-pill {{
    display: inline-block;
    padding: 1px 8px; border-radius: 999px;
    font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.4px; text-transform: uppercase;
    margin-right: 4px;
  }}
  .sb-pill-free   {{ background: #FEF3C7; color: #92400E; }}
  .sb-pill-starter{{ background: #DBEAFE; color: #1E40AF; }}
  .sb-pill-pro    {{ background: #E0E7FF; color: #3730A3; }}

  /* ---------- Chat-Input ---------- */
  div[data-testid="stChatInput"] {{
    border-radius: 16px !important;
    border: 1px solid {DUBLY_BORDER} !important;
    background: {DUBLY_CARD} !important;
    box-shadow: 0 4px 16px -4px rgba(10,10,15,0.06) !important;
  }}
  div[data-testid="stChatInput"] textarea {{
    border-radius: 14px !important;
    border: none !important;
    font-size: 15px !important;
  }}
  div[data-testid="stChatInput"] textarea:focus {{
    box-shadow: none !important;
  }}

  /* ---------- Footer ---------- */
  .dubly-footer {{
    margin-top: 28px; padding-top: 18px;
    border-top: 1px solid {DUBLY_BORDER};
    font-size: 11.5px; color: {DUBLY_SUBTLE};
    text-align: center; line-height: 1.6;
  }}
  .dubly-footer a {{ color: {DUBLY_MUTED}; text-decoration: none; }}
  .dubly-footer a:hover {{ color: {DUBLY_ACCENT}; }}
</style>
""", unsafe_allow_html=True)

# ---------- Hero ----------
st.markdown(f"""
<div class="dubly-hero">
  <div class="dubly-logo-row">
    <img class="dubly-logo" src="{DUBLY_LOGO_URL}" alt="Dubly.AI" />
    <span class="demo-badge"><span class="pulse"></span> Demo</span>
  </div>
  <h1>Hi, ich bin Dubly's<br/>KI-Support-Assistent.</h1>
  <p class="sub">
    Ich beantworte Fragen rund um Übersetzungen, Abo &amp; Pricing, schaue
    in deinen Account, und melde dich bei einem Menschen wenn's wichtig
    wird. Probier mich aus — alle Daten sind fiktiv.
  </p>
  <div class="trust-row">
    <span>✓ Made in Germany</span>
    <span class="dot"></span>
    <span>✓ GDPR-konform</span>
    <span class="dot"></span>
    <span>Session {_session_id()}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
MOCK_CUSTOMER_DISPLAY = [
    ("lisa.bauer@beispiel.de", "Lisa Bauer", "free", "Neu, 1 Test-Credit, keine Übersetzungen"),
    ("marco.rossi@example.com", "Marco Rossi", "free", "Test-Credit verbraucht, 1 Dub gemacht"),
    ("sarah.klein@beispiel.de", "Sarah Klein", "starter", "Paid, nur noch 8/100 Credits"),
    ("alex.chen@example.com", "Alex Chen", "pro", "Letzter Lip-Sync fehlgeschlagen"),
    ("marie.lefevre@example.com", "Marie Lefèvre", "pro", "Power-User, Job läuft gerade"),
]

with st.sidebar:
    st.markdown("### Was diese Demo kann")
    st.markdown(
        "Volles Bot-Verhalten mit Help-Center-Wissen, Account-Lookups, "
        "Plain-Tickets und Aktionen. Aktionen werden im Demo-Modus auto-bestätigt.",
        help="Im Produktiv-Bot würde jede Aktion eine Bestätigung erfordern.",
    )

    st.markdown("### Test-Kunden")
    for email, name, plan, descr in MOCK_CUSTOMER_DISPLAY:
        pill_class = f"sb-pill-{plan}"
        plan_label = {"free": "Free", "starter": "Starter", "pro": "Pro"}[plan]
        st.markdown(f"""
        <div class="sb-customer">
          <div><span class="sb-pill {pill_class}">{plan_label}</span> <strong>{name}</strong></div>
          <div class="sb-email">{email}</div>
          <div class="sb-meta">{descr}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Feedback")
    st.markdown(
        "Unter jeder Bot-Antwort findest du drei Knöpfe: 👍 hilfreich, "
        "👎 nicht hilfreich, 🐛 Fehler melden. Negatives Feedback landet "
        "zusätzlich als Thread in unserem Plain-Helpdesk."
    )

    st.markdown("### Aktionen")
    if st.button("Konversation zurücksetzen", use_container_width=True):
        for k in ["messages", "tool_trails", "feedback_open",
                  "feedback_given", "starter_clicked"]:
            st.session_state.pop(k, None)
        st.rerun()


# Session-State
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": ..., "content": ...}
if "tool_trails" not in st.session_state:
    st.session_state.tool_trails = {}   # message_index -> [trail entries]
if "feedback_open" not in st.session_state:
    st.session_state.feedback_open = {}   # message_index -> bool
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}   # message_index -> rating


def _render_msg(i: int, msg: dict) -> None:
    role = msg["role"]
    avatar = "🟧" if role == "assistant" else "🧑"
    with st.chat_message(role, avatar=avatar):
        content = msg["content"]
        if isinstance(content, list):
            # serialisierte Tool-Use-Blocks (kommen nicht ans UI)
            return
        st.markdown(content)
        # Tool-Trail
        trail = st.session_state.tool_trails.get(i, [])
        if trail and role == "assistant":
            with st.expander(f"Schritte des Bots ({len(trail)} Tool-Aufruf{'e' if len(trail)!=1 else ''})", expanded=False):
                for t in trail:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in t["args"].items())
                    st.markdown(f'<div class="tool-line"><strong>{t["tool"]}</strong>({args_str})</div>', unsafe_allow_html=True)
                    st.code(t["result_preview"], language="json")
        # Feedback (nur fuer assistant messages)
        if role == "assistant":
            already = st.session_state.feedback_given.get(i)
            cols = st.columns([1, 1, 1, 6])
            with cols[0]:
                if st.button("👍", key=f"fb_up_{i}", disabled=already is not None):
                    st.session_state.feedback_given[i] = "up"
                    st.session_state.feedback_open[i] = True
                    st.rerun()
            with cols[1]:
                if st.button("👎", key=f"fb_down_{i}", disabled=already is not None):
                    st.session_state.feedback_given[i] = "down"
                    st.session_state.feedback_open[i] = True
                    st.rerun()
            with cols[2]:
                if st.button("🐛", key=f"fb_bug_{i}", disabled=already is not None):
                    st.session_state.feedback_given[i] = "bug"
                    st.session_state.feedback_open[i] = True
                    st.rerun()
            with cols[3]:
                if already:
                    label = {"up": "👍 hilfreich", "down": "👎 nicht hilfreich", "bug": "🐛 Fehler"}[already]
                    st.caption(f"Feedback: {label}")
            if st.session_state.feedback_open.get(i, False):
                rating = st.session_state.feedback_given.get(i, "")
                comment = st.text_area(
                    "Kommentar (optional)",
                    key=f"fb_comment_{i}",
                    placeholder="Was war gut/schlecht? Was hat gefehlt? Was war falsch?",
                    height=68,
                )
                if st.button("Feedback senden", key=f"fb_send_{i}", type="primary"):
                    save_feedback(
                        message_id=str(i),
                        rating=rating,
                        comment=comment or "",
                        transcript=[m for m in st.session_state.messages if isinstance(m.get("content"), str)],
                    )
                    st.session_state.feedback_open[i] = False
                    st.toast("Danke für dein Feedback!")
                    st.rerun()


# Konfigurations-Check
if not ANTHROPIC_KEY:
    st.error("ANTHROPIC_API_KEY fehlt. In .env eintragen oder als Streamlit-Secret konfigurieren.")
    st.stop()
if not GEMINI_KEY:
    st.error("GEMINI_API_KEY fehlt (wird für Wissensbasis-Embeddings gebraucht).")
    st.stop()
if load_chunks()[0] is None:
    st.error("chunks_with_embeddings.json nicht gefunden — Wissensbasis fehlt.")
    st.stop()
if load_mock_db() is None:
    st.error("mock_customers.json nicht gefunden.")
    st.stop()


# ---------- Welcome State (nur wenn Konversation leer) ----------
STARTER_PROMPTS = [
    ("🎬", "Wie funktioniert die Trial?",
     "Wie funktioniert die kostenlose Trial bei Dubly?"),
    ("⚙️", "Mein Lip-Sync ist gefailed",
     "Hi, ich bin alex.chen@example.com — mein letzter Lip-Sync ist fehlgeschlagen. Was ist los?"),
    ("💳", "Was kosten die Pläne?",
     "Was kosten die verschiedenen Abo-Pläne? Was ist der Unterschied zwischen Starter und Pro?"),
    ("🚨", "Refund anfragen",
     "sarah.klein@beispiel.de — Ich möchte mein Abo kündigen und eine Rückerstattung."),
]

if not st.session_state.messages:
    st.markdown(
        f'<div class="welcome-eyebrow">Starte hier</div>'
        f'<h2 class="welcome-title">Worum geht es?</h2>'
        f'<p class="welcome-sub">'
        f"Klick auf eine typische Frage zum Reinkommen, oder schreib"
        f" einfach unten in den Chat. Du kannst dich als einer unserer"
        f" Test-Kunden ausgeben (Sidebar) — der Bot zeigt dir dann"
        f" personalisierte Antworten."
        f"</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="starter-grid">', unsafe_allow_html=True)
    cols = st.columns(2, gap="small")
    for i, (icon, title, prompt_text) in enumerate(STARTER_PROMPTS):
        with cols[i % 2]:
            if st.button(
                f"{icon}  {title}\n{prompt_text[:54]}{'…' if len(prompt_text)>54 else ''}",
                key=f"starter_{i}",
                use_container_width=True,
            ):
                st.session_state.starter_clicked = prompt_text
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Existierende Messages rendern
for i, msg in enumerate(st.session_state.messages):
    _render_msg(i, msg)

# Input — chat_input UND starter-clicked beide moeglich
prompt = st.chat_input("Stell deine Frage…")
if not prompt and "starter_clicked" in st.session_state:
    prompt = st.session_state.pop("starter_clicked")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    # Tool-Loop laufen lassen
    with st.chat_message("assistant", avatar="💬"):
        placeholder = st.empty()
        placeholder.markdown("_Einen Moment, ich schaue nach…_")
        # Konversation fuer API: nur strings als content
        api_conv = []
        for m in st.session_state.messages[:-1]:  # ohne letzte (das ist der Prompt)
            if isinstance(m["content"], str):
                api_conv.append({"role": m["role"], "content": m["content"]})
            else:
                api_conv.append({"role": m["role"], "content": m["content"]})
        try:
            final_text, trail = run_turn(api_conv, prompt)
        except Exception as e:  # noqa: BLE001
            final_text = f"_Fehler beim Aufruf: {type(e).__name__}: {e}_"
            trail = []
        placeholder.empty()
        st.session_state.messages.append({"role": "assistant", "content": final_text})
        new_idx = len(st.session_state.messages) - 1
        st.session_state.tool_trails[new_idx] = trail
    st.rerun()

# ---------- Footer ----------
st.markdown("""
<div class="dubly-footer">
  Powered by Claude · Built with the Claude Agent SDK<br/>
  Diese Demo verwendet ausschließlich fiktive Test-Daten.
  Feedback &amp; Bug-Reports landen direkt im Dubly-Helpdesk.
</div>
""", unsafe_allow_html=True)
