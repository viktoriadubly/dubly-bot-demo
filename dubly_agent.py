"""
Dubly Support-Agent (Stufe A)
==============================
Selbstbau-Plan 10b, Stufe A: Migration vom RAG-Bot auf den Claude Agent SDK.

Statt fest verdrahteter Pipeline (immer erst Wissensbasis abrufen, dann
antworten), bekommt Claude eine Liste von Tools und entscheidet selbst,
wann er sie aufruft. In Stufe A gibt es nur ein Tool:

  - search_knowledge_base(query, top_k)

In Stufe B kommen Plain- und Customer-Daten-Tools dazu, in Stufe C
Action-Tools mit Approval.

Voraussetzungen:
  - .env mit ANTHROPIC_API_KEY und GEMINI_API_KEY
  - chunks_with_embeddings.json (aus Stufe 1 vorhanden)
  - pip3 install claude-agent-sdk anthropic google-genai numpy

Ausfuehren:
    python3 dubly_agent.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
import uuid
import zoneinfo
from pathlib import Path


# Session-ID fuer Audit-Log (Stufe C)
SESSION_ID = uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# .env laden (mini-Parser, keine externe Lib noetig)
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


# ---------------------------------------------------------------------------
# Imports + Key-Checks
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import requests
    from google import genai
    from google.genai import types
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        tool,
        create_sdk_mcp_server,
        AssistantMessage,
        TextBlock,
        ToolUseBlock,
        ToolResultBlock,
        PermissionResultAllow,
        PermissionResultDeny,
    )
except ImportError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"[FEHLER] Bibliothek fehlt: {missing}")
    print("        Bitte einmal im Terminal:")
    print("            pip3 install claude-agent-sdk anthropic google-genai numpy requests")
    sys.exit(1)


ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
PLAIN_KEY = os.environ.get("PLAIN_API_KEY", "").strip()  # optional

if not ANTHROPIC_KEY:
    print("[FEHLER] ANTHROPIC_API_KEY fehlt in .env")
    sys.exit(2)
if not GEMINI_KEY:
    print("[FEHLER] GEMINI_API_KEY fehlt in .env (wird fuer Query-Embedding gebraucht)")
    sys.exit(2)
# PLAIN_KEY ist OPTIONAL -- ohne werden Plain-Tools mit klarer Meldung blockiert,
# der Rest des Bots laeuft normal weiter.


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
CHUNKS_FILE = Path(__file__).parent / "chunks_with_embeddings.json"
EMBEDDING_MODEL = "gemini-embedding-001"   # muss zu embed_chunks.py passen
CLAUDE_MODEL = "claude-haiku-4-5"           # Haiku: schnell + guenstig fuer Demo.
                                            # Fuer komplexere Konversationen spaeter
                                            # auf "claude-sonnet-4-6" hochstellen.
TOP_K_DEFAULT = 5
MIN_SIMILARITY = 0.55


# ---------------------------------------------------------------------------
# Wissensbasis laden
# ---------------------------------------------------------------------------
if not CHUNKS_FILE.exists():
    print(f"[FEHLER] {CHUNKS_FILE.name} nicht gefunden.")
    print("        Erst die Selbstbau-Stufe 1 ausfuehren:")
    print("            python3 scrape_helpcenter.py")
    print("            python3 embed_chunks.py")
    sys.exit(3)

print(f"Lade Wissensbasis aus {CHUNKS_FILE.name} ...")
with CHUNKS_FILE.open(encoding="utf-8") as f:
    CHUNKS = json.load(f)

EMBEDDINGS = np.array([c["embedding"] for c in CHUNKS], dtype=np.float32)
EMBEDDINGS = EMBEDDINGS / np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
print(f"  -> {len(CHUNKS)} Chunks bereit (Dim {EMBEDDINGS.shape[1]})")

GEMINI = genai.Client(api_key=GEMINI_KEY)


# ---------------------------------------------------------------------------
# Mock-Customer-DB laden (Stufe B Teil 1)
# ---------------------------------------------------------------------------
MOCK_DB_FILE = Path(__file__).parent / "mock_customers.json"
if not MOCK_DB_FILE.exists():
    print(f"[FEHLER] {MOCK_DB_FILE.name} nicht gefunden.")
    print("        Die Datei wird in Stufe B Teil 1 angelegt -- liegt normalerweise")
    print("        bereits im Projekt-Ordner.")
    sys.exit(4)

with MOCK_DB_FILE.open(encoding="utf-8") as f:
    _RAW = json.load(f)
CUSTOMERS = _RAW.get("customers", [])
_BY_EMAIL = {c["email"].lower(): c for c in CUSTOMERS}
_BY_ID = {c["id"]: c for c in CUSTOMERS}
_JOB_INDEX = {
    j["id"]: {"job": j, "customer": c}
    for c in CUSTOMERS for j in c.get("jobs", [])
}
print(f"  -> Mock-DB: {len(CUSTOMERS)} Test-Kunden geladen, "
      f"{len(_JOB_INDEX)} Jobs.")


def _customer_summary(c: dict) -> dict:
    """Public-facing customer summary (ohne nested credits/subscription).
    'language' aus dem Profil wird ABSICHTLICH NICHT zurueckgegeben, damit
    der Bot nicht versucht, den Chat-Sprach-Wechsel daran festzumachen --
    Chat-Sprache richtet sich immer nach der aktuellen User-Nachricht.
    """
    return {
        "id": c["id"],
        "email": c["email"],
        "name": c["name"],
        "joined_at": c.get("joined_at"),
        "plan": c.get("plan"),
    }


# ---------------------------------------------------------------------------
# Retrieval-Logik (wird vom Tool aufgerufen)
# ---------------------------------------------------------------------------
def _embed_query(text: str) -> np.ndarray:
    res = GEMINI.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    e = np.array(res.embeddings[0].values, dtype=np.float32)
    return e / np.linalg.norm(e)


def _retrieve(query: str, top_k: int) -> list[dict]:
    q_emb = _embed_query(query)
    sims = EMBEDDINGS @ q_emb
    idx = np.argsort(sims)[-top_k:][::-1]
    out = []
    for i in idx:
        c = CHUNKS[int(i)]
        out.append({
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "score": round(float(sims[i]), 3),
            "text": c.get("text", "")[:1200],  # kappen, um Token zu sparen
        })
    return out


# ---------------------------------------------------------------------------
# Tool-Definition (in-process MCP-Server)
# ---------------------------------------------------------------------------
@tool(
    "search_knowledge_base",
    "Durchsucht die Dubly-Help-Center-Artikel semantisch und gibt die "
    "Top-Treffer mit Quelle, Titel, Score und Volltext zurueck. Nutze dieses "
    "Tool fuer JEDE Sachfrage zu Dubly-Produkten, Features, Preisen, "
    "How-Tos, Fehlerbehebung. Nutze es NICHT fuer Smalltalk, Begruessungen "
    "oder offensichtlich nicht-Dubly-Themen. Wenn der hoechste Score < 0.55 "
    "ist, gilt: kein passender Artikel vorhanden -- ehrlich sagen und "
    "eskalieren statt zu raten.",
    {"query": str, "top_k": int},
)
async def search_knowledge_base(args: dict) -> dict:
    query = args.get("query", "").strip()
    top_k = int(args.get("top_k") or TOP_K_DEFAULT)
    if not query:
        return {"content": [{"type": "text", "text": "Fehler: leere Query."}]}
    top_k = max(1, min(top_k, 10))

    hits = _retrieve(query, top_k)
    top_score = hits[0]["score"] if hits else 0.0

    if top_score < MIN_SIMILARITY:
        body = (
            f"KEIN ARTIKEL UEBER SCHWELLE GEFUNDEN. "
            f"Bester Score: {top_score} (Schwelle: {MIN_SIMILARITY}).\n"
            "Das heisst: die Wissensbasis hat dazu nichts Passendes. "
            "Sei ehrlich, eskaliere statt zu raten.\n\n"
            "Hier trotzdem die naechsten Treffer zur Inspektion:\n\n"
        )
    else:
        body = f"GEFUNDEN: {len(hits)} Treffer (Top-Score {top_score}).\n\n"

    for i, h in enumerate(hits, start=1):
        body += (
            f"--- Treffer {i} (Score {h['score']}) ---\n"
            f"Titel: {h['title']}\n"
            f"Quelle: {h['url']}\n"
            f"{h['text']}\n\n"
        )

    return {"content": [{"type": "text", "text": body}]}


# ---------------------------------------------------------------------------
# Customer-Tools (Stufe B Teil 1, lesen die Mock-DB)
# ---------------------------------------------------------------------------
def _err(msg: str) -> dict:
    return {"content": [{"type": "text", "text": f"ERROR: {msg}"}]}


def _ok(payload) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}]}


@tool(
    "get_customer",
    "Schlaegt einen Kunden per Email-Adresse in der Customer-DB nach. "
    "Gibt id, name, plan, joined_at zurueck. Nutze dieses Tool als ERSTEN "
    "Schritt, sobald der User dir eine Email genannt hat oder ueber 'mein "
    "Account / meine Daten' spricht. Den zurueckgegebenen Wert 'id' brauchst "
    "du fuer alle weiteren Customer-Tools (get_subscription, get_credits, "
    "list_recent_jobs). Wenn kein Kunde gefunden wird: ehrlich sagen, NIEMALS "
    "Daten erfinden.",
    {"email": str},
)
async def get_customer(args: dict) -> dict:
    email = (args.get("email") or "").strip().lower()
    if not email:
        return _err("Parameter 'email' fehlt.")
    c = _BY_EMAIL.get(email)
    if not c:
        return _err(f"Kein Kunde mit Email '{email}' in der DB.")
    return _ok(_customer_summary(c))


@tool(
    "get_subscription",
    "Gibt die aktuelle Subscription des Kunden zurueck: plan_name, status, "
    "started_at, trial_ends_at (nur bei Trials), renewal_at (nur bei Paid), "
    "monthly_price_eur. Brauchst du customer_id von get_customer. Nutze "
    "dieses Tool bei Fragen zu Abo, Plan, Trial-Status, naechster Abrechnung, "
    "Kuendigung. Bei 'Kuendigung' VERWEISE auf einen Menschen (Eskalation), "
    "auch wenn du das Subscription-Datum siehst.",
    {"customer_id": str},
)
async def get_subscription(args: dict) -> dict:
    cid = (args.get("customer_id") or "").strip()
    if not cid:
        return _err("Parameter 'customer_id' fehlt.")
    c = _BY_ID.get(cid)
    if not c:
        return _err(f"Kein Kunde mit ID '{cid}' in der DB.")
    return _ok(c.get("subscription"))


@tool(
    "get_credits",
    "Gibt den aktuellen Credit-Stand zurueck: remaining (verfuegbar), "
    "monthly_total (Inklusivmenge des Plans), last_topup_at, expires_at. "
    "Brauchst du customer_id. Nutze dieses Tool bei jeder Frage zu Credits, "
    "Guthaben, 'wie viel hab ich noch', Verlauf der Aufladungen. Wenn "
    "remaining sehr niedrig ist (< 10% des monthly_total): proaktiv "
    "erwaehnen, statt nur die Zahl auszugeben.",
    {"customer_id": str},
)
async def get_credits(args: dict) -> dict:
    cid = (args.get("customer_id") or "").strip()
    if not cid:
        return _err("Parameter 'customer_id' fehlt.")
    c = _BY_ID.get(cid)
    if not c:
        return _err(f"Kein Kunde mit ID '{cid}' in der DB.")
    return _ok(c.get("credits"))


@tool(
    "list_recent_jobs",
    "Listet die letzten Jobs des Kunden, neueste zuerst. Jeder Job hat id, "
    "type (dub/lipsync/voice_clone), source_lang, target_lang, status "
    "(queued/running/completed/failed), created_at, duration_min, "
    "credits_used. Brauchst du customer_id. Nutze dieses Tool, wenn der "
    "User von 'meinem Job', 'meinem letzten Video' oder aehnlichem spricht "
    "OHNE eine konkrete Job-ID zu nennen -- du brauchst die richtige Job-ID "
    "fuer get_job_status.",
    {"customer_id": str, "limit": int},
)
async def list_recent_jobs(args: dict) -> dict:
    cid = (args.get("customer_id") or "").strip()
    limit = int(args.get("limit") or 5)
    if not cid:
        return _err("Parameter 'customer_id' fehlt.")
    c = _BY_ID.get(cid)
    if not c:
        return _err(f"Kein Kunde mit ID '{cid}' in der DB.")
    jobs = sorted(c.get("jobs", []), key=lambda j: j.get("created_at", ""), reverse=True)
    return _ok(jobs[: max(1, min(limit, 20))])


@tool(
    "get_job_status",
    "Holt den vollen Status eines bestimmten Jobs per job_id. Gibt alle "
    "Felder zurueck inkl. error_message bei failed Jobs und progress_pct "
    "bei running Jobs. Nutze dieses Tool, wenn du die job_id kennst (vom "
    "User direkt genannt ODER aus list_recent_jobs). Bei status='failed' "
    "schau dir error/error_message an, formuliere die Diagnose freundlich, "
    "und wenn du nicht weiterhelfen kannst: search_knowledge_base zum "
    "Error-Code -- und wenn auch das nichts liefert, eskaliere.",
    {"job_id": str},
)
async def get_job_status(args: dict) -> dict:
    jid = (args.get("job_id") or "").strip()
    if not jid:
        return _err("Parameter 'job_id' fehlt.")
    entry = _JOB_INDEX.get(jid)
    if not entry:
        return _err(f"Kein Job mit ID '{jid}' in der DB.")
    return _ok(entry["job"])


# ---------------------------------------------------------------------------
# Plain-Integration (Stufe B Teil 2)
# ---------------------------------------------------------------------------
PLAIN_API_URL = "https://core-api.uk.plain.com/graphql/v1"
PLAIN_TEST_PREFIX = "[BOT-TEST] "
# Optional: wenn du in Plain ein Label "bot-test" mit zugehoeriger labelTypeId
# hast, kannst du sie hier in .env eintragen (PLAIN_BOT_TEST_LABEL_ID=lt_...)
# und sie wird automatisch jedem Test-Thread angehaengt. Falls leer:
# wir setzen nur den Title-Prefix.
PLAIN_BOT_TEST_LABEL_ID = os.environ.get("PLAIN_BOT_TEST_LABEL_ID", "").strip()


def _plain_request(query: str, variables: dict) -> dict:
    """Sendet eine GraphQL-Mutation an Plain. Gibt JSON zurueck (oder Fehler-Dict)."""
    if not PLAIN_KEY:
        return {"_no_key": True}
    try:
        resp = requests.post(
            PLAIN_API_URL,
            headers={
                "Authorization": f"Bearer {PLAIN_KEY}",
                "Content-Type": "application/json",
            },
            json={"query": query, "variables": variables},
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        return {"errors": [{"message": f"Verbindungsfehler: {e}"}]}
    if resp.status_code != 200:
        return {"errors": [{"message": f"HTTP {resp.status_code}: {resp.text[:300]}"}]}
    return resp.json()


_PLAIN_UPSERT_CUSTOMER = """
mutation upsertCustomer($input: UpsertCustomerInput!) {
  upsertCustomer(input: $input) {
    result
    customer { id }
    error { message code }
  }
}
"""

_PLAIN_CREATE_THREAD = """
mutation createThread($input: CreateThreadInput!) {
  createThread(input: $input) {
    thread { id title }
    error { message code }
  }
}
"""

_PLAIN_CREATE_NOTE = """
mutation createNote($input: CreateNoteInput!) {
  createNote(input: $input) {
    note { id }
    error { message code }
  }
}
"""


def _plain_no_key_msg() -> dict:
    return _err(
        "Plain ist nicht konfiguriert (PLAIN_API_KEY fehlt in .env). "
        "Der Bot kann keine Tickets anlegen. Bitte sage dem User ehrlich, "
        "dass die Eskalation manuell erfolgen muss."
    )


def _plain_extract_thread_id(payload: dict) -> tuple[str | None, str | None]:
    if payload.get("errors"):
        return None, str(payload["errors"])[:300]
    data = (payload.get("data") or {}).get("createThread") or {}
    if data.get("error"):
        return None, str(data["error"])
    return ((data.get("thread") or {}).get("id"), None)


def _plain_upsert(email: str, name: str) -> tuple[str | None, str | None]:
    payload = _plain_request(_PLAIN_UPSERT_CUSTOMER, {
        "input": {
            "identifier": {"emailAddress": email},
            "onCreate": {
                "fullName": name,
                "email": {"email": email, "isVerified": False},
            },
            "onUpdate": {},
        }
    })
    if payload.get("_no_key"):
        return None, "Plain nicht konfiguriert"
    if payload.get("errors"):
        return None, str(payload["errors"])[:300]
    data = (payload.get("data") or {}).get("upsertCustomer") or {}
    if data.get("error"):
        return None, str(data["error"])
    return ((data.get("customer") or {}).get("id"), None)


def _plain_create_thread(customer_id: str, title: str, body: str) -> tuple[str | None, str | None]:
    full_title = title if title.startswith(PLAIN_TEST_PREFIX) else PLAIN_TEST_PREFIX + title
    inp: dict = {
        "title": full_title[:120],
        "customerIdentifier": {"customerId": customer_id},
        "components": [{"componentText": {"text": body[:9000]}}],
    }
    if PLAIN_BOT_TEST_LABEL_ID:
        inp["labelTypeIds"] = [PLAIN_BOT_TEST_LABEL_ID]
    payload = _plain_request(_PLAIN_CREATE_THREAD, {"input": inp})
    if payload.get("_no_key"):
        return None, "Plain nicht konfiguriert"
    return _plain_extract_thread_id(payload)


def _plain_add_note(thread_id: str, body: str) -> tuple[bool, str | None]:
    payload = _plain_request(_PLAIN_CREATE_NOTE, {
        "input": {"threadId": thread_id, "body": body[:9000]}
    })
    if payload.get("_no_key"):
        return False, "Plain nicht konfiguriert"
    if payload.get("errors"):
        return False, str(payload["errors"])[:300]
    data = (payload.get("data") or {}).get("createNote") or {}
    if data.get("error"):
        return False, str(data["error"])
    return True, None


@tool(
    "create_plain_thread",
    "Legt einen neuen Thread im Plain-Helpdesk fuer den genannten Kunden an. "
    "Nutze dieses Tool fuer normalere Faelle, in denen ein Mensch das Thema "
    "aufgreifen soll -- z.B. wenn der User um Rueckruf bittet, oder fuer Faelle "
    "die du nicht selbst loesen kannst aber die nicht so heiss sind, dass sofort "
    "eskaliert werden muss. Fuer dringende Eskalationen (Refund, Beschwerde, "
    "wuetende User) lieber escalate_to_human nehmen -- das setzt zusaetzlich "
    "die richtigen Labels und Note.",
    {"customer_email": str, "customer_name": str, "title": str, "summary": str},
)
async def create_plain_thread(args: dict) -> dict:
    if not PLAIN_KEY:
        return _plain_no_key_msg()
    email = (args.get("customer_email") or "").strip().lower()
    name = (args.get("customer_name") or "Dubly User").strip()
    title = (args.get("title") or "Bot-Konversation").strip()
    summary = (args.get("summary") or "").strip()
    if not email:
        return _err("customer_email fehlt.")

    cust_id, err = _plain_upsert(email, name)
    if err:
        return _err(f"Upsert fehlgeschlagen: {err}")

    thread_id, err = _plain_create_thread(cust_id, title, summary)
    if err:
        return _err(f"Thread-Erstellung fehlgeschlagen: {err}")
    return _ok({
        "thread_id": thread_id,
        "title_used": PLAIN_TEST_PREFIX + title[:120 - len(PLAIN_TEST_PREFIX)],
        "customer_id": cust_id,
        "label_attached": bool(PLAIN_BOT_TEST_LABEL_ID),
    })


@tool(
    "add_plain_note",
    "Haengt eine interne Note an einen Plain-Thread (sichtbar nur fuer Agenten, "
    "nicht fuer den Kunden). Nutze dieses Tool, um zusaetzlichen Kontext "
    "(Konversationsverlauf, technische Details, beobachtete Probleme) zu einem "
    "bestehenden Thread hinzuzufuegen. Du brauchst thread_id (von create_plain_thread "
    "oder escalate_to_human zurueckgegeben).",
    {"thread_id": str, "body": str},
)
async def add_plain_note(args: dict) -> dict:
    if not PLAIN_KEY:
        return _plain_no_key_msg()
    tid = (args.get("thread_id") or "").strip()
    body = (args.get("body") or "").strip()
    if not tid:
        return _err("thread_id fehlt.")
    if not body:
        return _err("body fehlt.")
    ok, err = _plain_add_note(tid, body)
    if not ok:
        return _err(f"Note konnte nicht gesetzt werden: {err}")
    return _ok({"status": "note_added", "thread_id": tid})


@tool(
    "escalate_to_human",
    "Eskaliert die aktuelle Konversation an einen Menschen im Plain-Helpdesk. "
    "Erstellt einen Test-Thread mit [BOT-TEST]-Prefix im Titel, haengt eine "
    "Note mit der vollen Begruendung dran, und gibt thread_id zurueck. Nutze "
    "dies bei: Refund/Cancel/Beschwerde-Triggern, ausdruecklichem Wunsch nach "
    "Mensch, technischen Bugs die Account-Zugriff brauchen, emotional "
    "aufgeladenen Usern, Faellen wo du nicht weiterhelfen kannst. "
    "Im 'summary' sollte stehen: wer der User ist (Name/Email/Plan wenn "
    "bekannt), was er will, was du bisher versucht hast, was der konkrete "
    "Eskalations-Grund ist. Nach erfolgreichem Tool-Call dem User mitteilen: "
    "Thread-ID, wann das Team antwortet (siehe SLA-Block), was er tun kann "
    "wenn nichts kommt.",
    {"customer_email": str, "customer_name": str, "reason": str, "summary": str},
)
async def escalate_to_human(args: dict) -> dict:
    if not PLAIN_KEY:
        return _plain_no_key_msg()
    email = (args.get("customer_email") or "").strip().lower()
    name = (args.get("customer_name") or "Dubly User").strip()
    reason = (args.get("reason") or "Bot-Eskalation").strip()
    summary = (args.get("summary") or "").strip()
    if not email:
        return _err("customer_email fehlt.")
    if not summary:
        return _err("summary fehlt (Kontext fuer den menschlichen Agenten).")

    cust_id, err = _plain_upsert(email, name)
    if err:
        return _err(f"Upsert fehlgeschlagen: {err}")

    title = f"Escalation: {reason}"
    body = (
        f"BOT-ESKALATION (Selbstbau-Bot, Stufe B Teil 2)\n\n"
        f"Kunde: {name} <{email}>\n"
        f"Grund: {reason}\n\n"
        f"Zusammenfassung vom Bot:\n{summary}\n"
    )
    thread_id, err = _plain_create_thread(cust_id, title, body)
    if err:
        return _err(f"Thread-Erstellung fehlgeschlagen: {err}")

    # Eskalations-Note (intern) draufpacken
    note_body = (
        f"Bot-Eskalations-Tag: ESCALATE\n"
        f"Grund: {reason}\n"
        f"Customer: {name} <{email}>\n\n"
        f"{summary}"
    )
    ok, err = _plain_add_note(thread_id, note_body)
    note_status = "added" if ok else f"failed: {err}"

    return _ok({
        "status": "escalated",
        "thread_id": thread_id,
        "title_used": PLAIN_TEST_PREFIX + title[:120 - len(PLAIN_TEST_PREFIX)],
        "customer_id": cust_id,
        "note": note_status,
        "label_attached": bool(PLAIN_BOT_TEST_LABEL_ID),
    })


# ---------------------------------------------------------------------------
# Action-Tools + Audit-Log (Stufe C)
# ---------------------------------------------------------------------------
AUDIT_LOG_FILE = Path(__file__).parent / "audit_log.jsonl"


def _persist_mock_db() -> None:
    """Schreibt die in-memory Customer-DB zurueck in mock_customers.json."""
    payload = dict(_RAW)
    payload["customers"] = CUSTOMERS
    MOCK_DB_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _audit(tool_name: str, args: dict, status: str, detail: dict | None = None) -> None:
    """Schreibt einen Eintrag in audit_log.jsonl."""
    entry = {
        "ts": dt.datetime.utcnow().isoformat() + "Z",
        "session_id": SESSION_ID,
        "tool": tool_name,
        "args": args,
        "status": status,           # "executed" | "denied" | "failed"
        "detail": detail or {},
    }
    with AUDIT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@tool(
    "grant_test_credits",
    "Schreibt einem Free-User in AUSNAHMEFAELLEN zusaetzliche Test-Credits "
    "gut (max 5 pro Aufruf, hartes Code-Limit). Dubly-Geschaeftsmodell "
    "(Test-Credit-Menge, Plan-Optionen, Pricing) findest du im Help Center "
    "ueber search_knowledge_base -- nutze das, statt aus dem Gedaechtnis "
    "Zahlen zu nennen. "
    ""
    "NUTZE dieses Tool NUR wenn EINER dieser triftigen Gruende vorliegt: "
    "(a) User berichtet glaubhaft von Bug/Plattform-Fehler, der seinen "
    "Test-Credit ohne brauchbares Ergebnis verbraucht hat. "
    "(b) Der einzige Test des Users ist durch UNSER Verschulden gescheitert "
    "(Silent Failure, Service-Ausfall, Worker-Crash). NICHT bei Input-"
    "Fehlern wie audio_quality_too_low -- die liegen am User. "
    "(c) Anderer klar dokumentierbarer Grund (in reason festhalten und "
    "im Zweifel an Mensch eskalieren statt selbst geben). "
    ""
    "NUTZE dieses Tool NICHT wenn der User einfach: 'mehr testen will' / "
    "'noch andere Sprachen probieren' / 'unsicher ist ob er abonnieren "
    "soll'. Das sind Sales-Objections, keine Kulanz-Gruende. Bei solchen "
    "Anfragen verweise auf die Abo-Optionen aus dem Help Center. "
    ""
    "NUR fuer plan=free; bei paid-Kunden NICHT verwenden (die wuerdest "
    "du mit apply_credit_bonus bedienen).",
    {"customer_id": str, "credits": int, "reason": str},
)
async def grant_test_credits(args: dict) -> dict:
    cid = (args.get("customer_id") or "").strip()
    try:
        credits = int(args.get("credits") or 0)
    except (TypeError, ValueError):
        _audit("grant_test_credits", args, "failed", {"reason": "invalid credits"})
        return _err("credits muss eine Zahl sein.")
    reason = (args.get("reason") or "").strip()
    if not cid:
        _audit("grant_test_credits", args, "failed", {"reason": "no customer_id"})
        return _err("customer_id fehlt.")
    if not 1 <= credits <= 5:
        _audit("grant_test_credits", args, "failed", {"reason": "credits out of range"})
        return _err("credits muss zwischen 1 und 5 liegen.")
    if not reason:
        _audit("grant_test_credits", args, "failed", {"reason": "no reason"})
        return _err("reason fehlt (Begruendung pflicht fuers Audit-Log).")
    c = _BY_ID.get(cid)
    if not c:
        _audit("grant_test_credits", args, "failed", {"reason": "unknown customer"})
        return _err(f"Kein Kunde mit ID '{cid}'.")
    if c.get("plan") != "free":
        _audit("grant_test_credits", args, "failed",
               {"reason": "not free user", "current_plan": c.get("plan")})
        return _err(
            f"Kunde ist auf Plan '{c.get('plan')}' -- grant_test_credits ist "
            "nur fuer Free-User gedacht. Bei Paid-Kunden apply_credit_bonus nehmen."
        )
    cr = c.get("credits") or {}
    old = int(cr.get("remaining", 0))
    new = old + credits
    cr["remaining"] = new
    _persist_mock_db()
    _audit("grant_test_credits", args, "executed",
           {"customer_id": cid, "old_remaining": old, "new_remaining": new,
            "credits_added": credits, "reason": reason})
    return _ok({
        "status": "test_credits_granted",
        "customer_id": cid,
        "old_remaining": old,
        "new_remaining": new,
        "credits_added": credits,
    })


@tool(
    "restart_lipsync_job",
    "Startet einen Job neu, der aktuell den Status 'failed' hat. Job-Typ "
    "egal (dub/lipsync/voice_clone). Nutze dies, wenn ein User von einem "
    "fehlgeschlagenen Job berichtet UND du den Eindruck hast, das war ein "
    "voruebergehender Fehler (z.B. Worker-Stau, Timeout). NICHT nutzen bei "
    "Input-Fehlern wie audio_quality_too_low -- da muss der User selbst "
    "etwas tun (z.B. lauteres Audio hochladen). Du SCHLAEGST den Neustart "
    "vor, Bestaetigung im Terminal.",
    {"job_id": str},
)
async def restart_lipsync_job(args: dict) -> dict:
    jid = (args.get("job_id") or "").strip()
    if not jid:
        _audit("restart_lipsync_job", args, "failed", {"reason": "no job_id"})
        return _err("job_id fehlt.")
    entry = _JOB_INDEX.get(jid)
    if not entry:
        _audit("restart_lipsync_job", args, "failed", {"reason": "unknown job"})
        return _err(f"Kein Job mit ID '{jid}'.")
    job = entry["job"]
    if job.get("status") != "failed":
        _audit("restart_lipsync_job", args, "failed",
               {"reason": "not failed", "current_status": job.get("status")})
        return _err(
            f"Job {jid} hat Status '{job.get('status')}', nur 'failed' Jobs "
            "koennen neu gestartet werden."
        )
    old_status = job["status"]
    job["status"] = "queued"
    job.pop("error", None)
    job.pop("error_message", None)
    job["restarted_at"] = dt.datetime.utcnow().isoformat() + "Z"
    _persist_mock_db()
    _audit("restart_lipsync_job", args, "executed",
           {"job_id": jid, "old_status": old_status, "new_status": "queued"})
    return _ok({
        "status": "restarted",
        "job_id": jid,
        "old_status": old_status,
        "new_status": "queued",
    })


@tool(
    "apply_credit_bonus",
    "Schreibt dem Kunden Kulanz-Credits gut (max 20 pro Aufruf). Nutze dies "
    "als Wiedergutmachung bei klaren Bot-/Plattform-Fehlern (failed Job durch "
    "uns, Service-Ausfall, Bug). NICHT als Verkaufsanreiz oder Trial-"
    "Verlaengerungs-Ersatz. Im Tool-Argument 'reason' kurz dokumentieren, "
    "warum die Kulanz gerechtfertigt ist -- landet im Audit-Log.",
    {"customer_id": str, "credits": int, "reason": str},
)
async def apply_credit_bonus(args: dict) -> dict:
    cid = (args.get("customer_id") or "").strip()
    try:
        credits = int(args.get("credits") or 0)
    except (TypeError, ValueError):
        _audit("apply_credit_bonus", args, "failed", {"reason": "invalid credits"})
        return _err("credits muss eine Zahl sein.")
    reason = (args.get("reason") or "").strip()
    if not cid:
        _audit("apply_credit_bonus", args, "failed", {"reason": "no customer_id"})
        return _err("customer_id fehlt.")
    if not 1 <= credits <= 20:
        _audit("apply_credit_bonus", args, "failed", {"reason": "credits out of range"})
        return _err("credits muss zwischen 1 und 20 liegen.")
    if not reason:
        _audit("apply_credit_bonus", args, "failed", {"reason": "no reason"})
        return _err("reason fehlt (Begruendung pflicht fuers Audit-Log).")
    c = _BY_ID.get(cid)
    if not c:
        _audit("apply_credit_bonus", args, "failed", {"reason": "unknown customer"})
        return _err(f"Kein Kunde mit ID '{cid}'.")
    cr = c.get("credits") or {}
    old = int(cr.get("remaining", 0))
    new = old + credits
    cr["remaining"] = new
    _persist_mock_db()
    _audit("apply_credit_bonus", args, "executed",
           {"customer_id": cid, "old_remaining": old, "new_remaining": new,
            "credits_added": credits, "reason": reason})
    return _ok({
        "status": "credits_applied",
        "customer_id": cid,
        "old_remaining": old,
        "new_remaining": new,
        "credits_added": credits,
    })


# Set der Tools, die durch den Approval-Callback gehen muessen:
ACTION_TOOL_NAMES = {
    "mcp__dubly__grant_test_credits",
    "mcp__dubly__restart_lipsync_job",
    "mcp__dubly__apply_credit_bonus",
}


async def approval_callback(tool_name: str, input_data: dict, context) -> "PermissionResultAllow | PermissionResultDeny":
    """Wird vor jedem Tool-Aufruf vom SDK gerufen. Bei Action-Tools im
    Terminal nach y/n fragen, sonst durchwinken."""
    if tool_name not in ACTION_TOOL_NAMES:
        return PermissionResultAllow(
            behavior="allow", updated_input=input_data, updated_permissions=None,
        )
    short = tool_name.split("__")[-1]
    args_str = ", ".join(f"{k}={v!r}" for k, v in input_data.items())
    print()
    print(f"  [APPROVAL] Bot moechte ausfuehren:")
    print(f"             {short}({args_str})")
    answer = await asyncio.to_thread(input, "  Ausfuehren? [y/n]: ")
    if answer.strip().lower() in {"y", "yes", "j", "ja"}:
        print("  -> APPROVED")
        return PermissionResultAllow(
            behavior="allow", updated_input=input_data, updated_permissions=None,
        )
    _audit(short, input_data, "denied",
           {"reason": answer.strip() or "user declined"})
    print("  -> DENIED")
    return PermissionResultDeny(
        behavior="deny",
        message="User declined the action via terminal approval.",
        interrupt=False,
    )


# ---------------------------------------------------------------------------
# SLA-Logik (1:1 aus dubly_bot_rag.py)
# ---------------------------------------------------------------------------
DUBLY_TZ = zoneinfo.ZoneInfo("Europe/Berlin")
WORKDAY_END_HOUR = 17


def build_sla_context() -> str:
    now = dt.datetime.now(DUBLY_TZ)
    weekday = now.weekday()
    is_weekend = weekday >= 5
    is_after_hours = now.hour >= WORKDAY_END_HOUR
    if is_weekend:
        situation = "Wochenende"
        normal_reply = ("'We'll come back to you next business day morning.' / "
                        "'Wir melden uns am naechsten Werktag frueh.'")
        critical_reply = ("Fuer kritische Themen: 'I'm flagging this to our on-call team "
                          "right now -- you'll hear back as fast as possible.'")
    elif is_after_hours:
        situation = "Werktag nach 17 Uhr"
        normal_reply = ("'We'll come back to you tomorrow morning.' / "
                        "'Wir melden uns morgen frueh bei dir.'")
        critical_reply = normal_reply
    else:
        situation = "Werktag (Mo-Fr) vor 17 Uhr"
        normal_reply = ("'Our team will come back to you with an update today.' / "
                        "'Wir melden uns heute noch bei dir.'")
        critical_reply = normal_reply
    return (
        f"# AKTUELLE ZEIT-INFO (fuer SLA bei Eskalationen)\n"
        f"- Jetzt: {now.strftime('%A, %d.%m.%Y %H:%M')} (Europe/Berlin)\n"
        f"- Situation: {situation}\n"
        f"- Normale Anfrage  -> {normal_reply}\n"
        f"- Kritische Anfrage -> {critical_reply}"
    )


# ---------------------------------------------------------------------------
# System-Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BASE = """Du bist der KI-Support-Assistent von Dubly.AI -- einer SaaS-Plattform fuer KI-Video-Dubbing. Du sprichst direkt mit Kunden im Chat.

# SPRACH-REGEL (WICHTIGSTE REGEL, NIEMALS BRECHEN)
Antworte IMMER in der Sprache der LETZTEN User-Nachricht.
- User schreibt Deutsch -> du antwortest Deutsch.
- User schreibt Englisch -> du antwortest Englisch.
- User wechselt mitten in der Konversation -> du wechselst sofort mit.

NICHT relevant fuer die Wahl der Antwortsprache: Email-Domain (.de/.com),
Name des Kunden, irgendein Profil-Feld, was du in einem frueheren Turn
geantwortet hast. NUR die aktuelle Nachricht zaehlt.

Beispiele:
- "Hi, wie viele Credits habe ich?"      -> Deutsch antworten
- "Hi, how many credits do I have?"      -> Englisch antworten
- "alex.chen@example.com -- wie geht's?" -> Deutsch antworten (User schrieb DE!)
- "lisa.bauer@beispiel.de, what's left?" -> Englisch antworten (User schrieb EN!)

# DEIN CHARAKTER (Voice Guide)
- Geduzt im Deutschen, "you" im Englischen. Niemals "Sie".
- Warm und kompetent, aber nicht kumpelhaft. "Hi!" als Einstieg ist gut.
- Kurz im Chat: 1-3 Saetze ODER eine praezise Rueckfrage.
- Emojis sparsam: max. 1 pro Antwort.

# TERMINOLOGIE (WICHTIG, IMMER ANWENDEN)
Die Tools sprechen intern von "Jobs" (z.B. list_recent_jobs, get_job_status,
job_id). DAS IST ENTWICKLER-SPRACHE. Im Chat mit dem User sagst du
NIEMALS "Job". Stattdessen, je nach Kontext:
- Allgemein: "dein Video" / "your video" oder "deine Uebersetzung" /
  "your translation"
- Wenn type=dub:        "dein Dub" / "your dub"
- Wenn type=lipsync:    "dein Lip-Sync" / "your lip sync"
- Wenn type=voice_clone: "deine Stimm-Klone" / "your voice clone"

Beispiele:
- Schlecht: "Dein letzter Job ist fehlgeschlagen."
- Gut:     "Dein letzter Lip-Sync ist fehlgeschlagen."
- Schlecht: "Ich starte den Job neu."
- Gut:     "Ich starte den Lip-Sync neu." / "Ich starte das Video neu."

# DEINE WERKZEUGE
Du hast NEUN Tools. Waehle das richtige fuer die Situation:

WISSEN
1. search_knowledge_base(query, top_k)
   -> Fuer Sachfragen zu Dubly-Produkten/Features/Preisen/How-Tos.
   -> Beispiele: "How do I export SRT?", "Welche Sprachen gibt's?"

CUSTOMER-DATEN (Mock-DB)
2. get_customer(email)
   -> Erster Schritt sobald Email genannt wurde oder es um "meinen Account" geht.
   -> Liefert customer_id, die du fuer alle weiteren Customer-Tools brauchst.
3. get_subscription(customer_id)
   -> Fuer Fragen zu Abo, Plan, Trial-Ende, Renewal.
4. get_credits(customer_id)
   -> Fuer Fragen zu Credit-Stand, Guthaben, "wie viel hab ich noch".
5. list_recent_jobs(customer_id, limit)
   -> Wenn der User von "meinem Job/Video" OHNE konkrete ID spricht.
6. get_job_status(job_id)
   -> Volldetails eines Jobs. Bei failed: error_message lesen, mit
      search_knowledge_base nach Loesung suchen.

PLAIN-HELPDESK (Echte Tickets)
7. create_plain_thread(customer_email, customer_name, title, summary)
   -> Wenn ein Mensch das Thema aufgreifen soll, aber's nicht heiss/dringend
      ist (Rueckruf-Bitte, Feature-Anfrage, "kann jemand das pruefen").
8. add_plain_note(thread_id, body)
   -> Zusatz-Kontext an einen bereits erstellten Thread haengen.
9. escalate_to_human(customer_email, customer_name, reason, summary)
   -> Fuer DRINGENDE Eskalationen: Refund/Cancel/Beschwerde-Triggers,
      "ich will einen Menschen", emotional aufgeladene User, Account-
      Aktionen, Bugs die Account-Zugriff brauchen.
   -> Erstellt Thread + Eskalations-Note in einem Schritt.

ACTION-TOOLS (Approval-pflichtig! Nutzer/Agent bestaetigt im Terminal)
10. grant_test_credits(customer_id, credits, reason)
    -> 1-5 zusaetzliche Test-Credits fuer FREE-User (Dubly-Modell: neue
       User bekommen 1 Test-Credit, danach Abo). Nutze, wenn ein neuer
       User noch zoegert oder seinen ersten Test verpatzt hat.
    -> NICHT fuer Paid-User (Starter/Pro) -- da apply_credit_bonus nehmen.
11. restart_lipsync_job(job_id)
    -> Nur fuer Jobs mit Status 'failed'. NICHT bei Input-Fehlern wie
       audio_quality_too_low (da muss der User selbst was tun).
    -> Nutze bei vermutetem temporaeren Fehler (Worker-Stau, Timeout).
12. apply_credit_bonus(customer_id, credits, reason)
    -> Max 20 Credits Kulanz. Fuer PAID-User bei klaren Bot-/Plattform-
       Fehlern (failed Job durch uns, Service-Ausfall, Bug).
    -> reason muss kurz die Rechtfertigung benennen (landet im Audit-Log).

# UNTERSCHIED grant_test_credits vs apply_credit_bonus
- grant_test_credits: Free-User probiert noch, will mehr Test-Credits. Marketing-
  /Conversion-Use-Case. Limit 5.
- apply_credit_bonus: Paid-Kunde wurde von uns enttaeuscht (Bug, Fehler).
  Goodwill-Use-Case. Limit 20.
Niemals beide auf denselben User -- der Plan entscheidet welches.

# ACTION-TOOL-REGELN (WICHTIG)

KRITISCH: Frage NIE rhetorisch "darf ich?" / "soll ich?" / "ist das okay?"
in DEM SELBEN Turn in dem du das Tool aufrufst. Das klingt komisch, weil die
echte Bestaetigung separat im Terminal passiert -- die Frage haengt dann
ohne Antwort in der Luft.

Zwei zulaessige Muster:

MUSTER A (Default): Kurze ANKUENDIGUNG + Tool-Call im selben Turn.
  - Du sagst in EINEM Satz, was du jetzt tust (Indikativ, kein Fragezeichen).
  - Dann ruf das Tool auf.
  - Im Terminal taucht das [APPROVAL]-Prompt auf, das ist die echte Bestaetigung.
  - Beispiel-Ankuendigung: "Ich gebe dir 3 Test-Credits, damit du weitere
    Sprachen ausprobieren kannst."
  - Schlecht: "Soll ich dir 3 Test-Credits geben?" (Frage + Tool in 1 Turn)

MUSTER B (Selten, wenn UNKLAR was/wieviel): Rueckfrage + Turn beenden.
  - Du fragst, was der User braucht (z.B. "Wieviele Credits sollen es sein?")
  - KEIN Tool-Call in diesem Turn.
  - Naechster Turn: User antwortet, DU rufst Tool auf mit Muster A.

Weitere Regeln:
1. Nach Ausfuehrung: Ergebnis dem User bestaetigen, in seiner Sprache.
   Voice-Guide-Regel gilt (warm, kompetent).
2. KEINE reflexiven Bestaetigungs-Ausrufe als Eroeffnung. Verboten:
   "Fertig!", "Done!", "Perfekt!", "Super!", "Klar!", "Gerne!", "Erledigt!",
   "Voila!" -- weder am Anfang noch isoliert. Geh DIREKT zur Sache:
   Schlecht: "Fertig! Du hast jetzt 10 Test-Credits."
   Gut:     "Du hast jetzt 10 Test-Credits."
   Schlecht: "Perfekt! Trial verlaengert um 7 Tage."
   Gut:     "Trial verlaengert um 7 Tage."
3. Bei Decline (User sagt n / decline / nein im Terminal): freundlich
   akzeptieren, alternative Hilfe anbieten. KEINE Penetranz, kein zweites
   Aufrufen.
4. NIEMALS so tun als haettest du eine Aktion ausgefuehrt, die nicht
   approved wurde. Du siehst im Tool-Result, ob "executed" oder "denied".

# WER ENTSCHEIDET UEBER AKTIONEN (WICHTIG)
DU bist der Gatekeeper, nicht der Servierer. Wenn ein User um eine Action
bittet ("gib mir Credits", "verlaengere", "starte neu"), entscheidest DU
ob das gerechtfertigt ist -- basierend auf der Datenlage und den Tool-
Regeln. Eine User-Bitte allein ist KEIN Grund eine Action durchzufuehren.

Faustregel fuer grant_test_credits: bei reiner Anfrage "kann ich mehr
testen" / "ich will noch mehr probieren bevor ich abonniere" sagst du
freundlich NEIN und verweist auf die Abo-Optionen aus dem Help Center.
Nur bei nachvollziehbarem PLATTFORM-Fehler von uns gibst du Kulanz-Credits.

# QUELLE FUER GESCHAEFTS-FAKTEN
Konkrete Geschaefts-Fakten (Pricing, Plan-Inhalt, Test-Credit-Menge,
Refund-Bedingungen, unterstuetzte Sprachen, Feature-Listen, AGB-Themen)
NIEMALS aus dem Gedaechtnis nennen -- IMMER ueber search_knowledge_base
holen. Das Help Center ist die einzige Quelle der Wahrheit.

Beispiele:
- "Was kostet der Pro-Plan?"        -> search_knowledge_base, nicht raten.
- "Wie viele Credits brauche ich
   fuer 5 Minuten Dub?"              -> search_knowledge_base.
- "Kriege ich Geld zurueck?"        -> search_knowledge_base zu Refund-
                                       Policy, dann ggf. Eskalation.
- "Unterstuetzt ihr Arabisch?"      -> search_knowledge_base.

Wenn der Treffer im Help Center nichts liefert (Score < 0.55), sag ehrlich
"weiss ich nicht" und eskaliere -- KEINE Zahlen erfinden.

# TOOL-NUTZUNGS-REGELN
- Formuliere Queries und Lookups in der gleichen Sprache wie der User.
- Bei mehrteiligen Fragen ggf. mehrere Tools nacheinander aufrufen.
- Ruf Tools direkt auf -- KEIN "Moment, ich schaue nach" vorher schreiben.
  Dem User wird der Tool-Aufruf separat angezeigt.
- Wenn ein Tool nichts findet (kein Kunde, kein Job, kein Treffer): ehrlich
  sagen, NIEMALS Daten erfinden.

# IDENTITY-CHECK (NUR wenn unbedingt noetig)
Standardvorgehen: bei JEDER Anfrage zuerst search_knowledge_base versuchen,
ohne nach Email zu fragen. Das Help Center loest etwa 80% der Anfragen.

Email/Identity NUR fragen wenn du eines dieser Tools brauchst:
- get_customer / get_subscription / get_credits
- list_recent_jobs / get_job_status
- grant_test_credits / restart_lipsync_job / apply_credit_bonus

NICHT nach Email fragen bei generischen Anfragen wie:
- How-Tos ("Wie exportiere ich als SRT?")
- Troubleshooting-Fragen die kein Account-Lookup brauchen
  ("Mein Video hat Audio-Artefakte" -> Help Center hat die Loesung)
- Pricing/Plan-Fragen
- Feature-Fragen

Job-ID NUR erfragen wenn nach Email-Lookup via list_recent_jobs kein
eindeutiger Job auffindbar ist -- also fast nie.

# ENTSCHEIDUNGS-LOGIK PRO TURN
1. Generische Frage (How-To, Trouble, Pricing, Feature)?
   -> search_knowledge_base direkt, antworten. KEINE Identity-Frage.
2. User spricht klar ueber SEINEN Account und Email fehlt?
   -> Hoeflich nach Email fragen.
3. Email schon im Verlauf -> get_customer und passende Tools.
4. Help Center liefert nichts (Score < 0.55) -> ehrlich, eskalieren.

Selbstauskunft per Email akzeptieren wir in dieser Demo-Stufe (in Produktion
spaeter via Session-Token).

# RUECKFRAGE ODER DIREKTE ANTWORT?
Bevor du eine Antwort gibst, pruefe:
1. Ist die Frage eindeutig? (Welches Video? Welcher Plan? Welche Sprache?)
2. Habe ich genug Kontext fuer eine PRAEZISE Antwort?
3. Koennte meine Antwort wegen fehlender Info am Beduerfnis vorbeigehen?

Wenn nein -> stell EINE einzige praezise Rueckfrage. Niemals 3 Fragen auf einmal.

## Beispiele fuer gute Rueckfragen
User: "Mein Video laedt nicht hoch."
Bot: "Mist! Damit ich helfen kann: Bekommst du eine Fehlermeldung, oder haengt es einfach bei einem bestimmten Prozentsatz?"

User: "How long does dubbing take?"
Bot: "Depends on your video length and the features you use -- are you asking for a quick estimate for a specific video, or in general?"

# ZWEI GOLDENE REGELN

## Regel #1: Niemals raten
Wenn das Tool keinen passenden Artikel liefert (Score < 0.55, oder das Tool sagt "kein Treffer"), sage ehrlich "Ich bin mir nicht sicher" und eskaliere. NIEMALS plausibel klingende Schritte erfinden.

## Regel #2: Bei Eskalationen verbindlich sein
Drei konkrete Sachen: was du tust, wann der Kunde Antwort bekommt, was er tun kann falls Antwort ausbleibt. NIE "as soon as possible". Die richtige Zeitzusage haengt von der Uhrzeit ab -- siehe SLA-Block unten.

# ESKALATIONS-TRIGGER (sofort an Mensch)
- Woerter wie refund, cancel, scammed, kuendigen, Beschwerde
- Geldbetraege ueber 50 EUR / 50 USD
- Emotional aufgeladene Kunden
- Kunde fragt nach Mensch
- Bei zwei Versuchen nicht klar verstanden
- Account-Aktionen (loeschen, Plan aendern)
- Spezifische Bug-Reports die Account-Zugriff brauchen

ESKALATIONS-ABLAUF (WICHTIG, hat sich in Stufe B Teil 2 geaendert):

Wenn ein Trigger zutrifft, mach das in genau dieser Reihenfolge:
1. Wenn moeglich, hol Identitaet/Kontext per get_customer/get_subscription
   (NUR wenn du Email hast, sonst frag kurz nach).
2. Rufe escalate_to_human auf mit:
   - customer_email = die Email die du hast
   - customer_name  = der Name aus get_customer (falls bekannt, sonst Email vor @)
   - reason         = kurzer Grund, z.B. "Refund request", "Cancellation"
   - summary        = 2-4 Saetze: Was der User will, was du bisher weisst,
                      ob es dringend wirkt
3. Nimm aus der Tool-Antwort die thread_id und sage sie dem User
   ("Ich habe deinen Fall unter Ticket {thread_id} angelegt").
4. Schreibe in deiner Antwort an den User: was du getan hast (Ticket
   angelegt), wann das Team sich meldet (siehe SLA-Block), was er tun
   kann wenn sich nichts ruehrt.
5. Optional: schreibe in eigener Zeile am Ende noch [ESCALATE] -- nur
   damit du es im Terminal sichtbar machst. KEIN Pflicht-Tag mehr.

Wenn PLAIN_API_KEY nicht konfiguriert ist (Tool meldet das), sag dem User
ehrlich, dass du das Ticket gerade nicht anlegen kannst und welchen Weg er
direkt nehmen soll (z.B. Email an support@dubly.ai).

# QUELLEN
Wenn du das Tool genutzt und passende Artikel gefunden hast, nenne am Ende deiner Antwort die genutzte Quell-URL in Klammern. Beispiel: (Quelle: https://support.dubly.ai/...)
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_BASE + "\n\n" + build_sla_context()


# ---------------------------------------------------------------------------
# Hauptschleife (Terminal-Chat)
# ---------------------------------------------------------------------------
def _build_greeting() -> str:
    plain_status = "AKTIV (echte Tickets!)" if PLAIN_KEY else "INAKTIV (kein PLAIN_API_KEY)"
    return (
        "\n========================================================\n"
        " Dubly Support-Agent (Stufe C, Agent-SDK)\n"
        "========================================================\n"
        f" Plain-Integration:  {plain_status}\n"
        f" Action-Tools:       AKTIV (Approval im Terminal mit y/n)\n"
        f" Session-ID:         {SESSION_ID}\n"
        f" Audit-Log:          audit_log.jsonl\n"
        " Tipp: Probier Action-Faelle, der Bot fragt jeweils nach Approval.\n"
        "       Beispiele:\n"
        "         - lisa.bauer@beispiel.de, kann ich noch mehr testen bevor ich abonniere?\n"
        "         - marco.rossi@example.com, mein Test-Video ist nicht gut geworden :(\n"
        "         - alex.chen@example.com, my lipsync failed, please retry\n"
        "         - sarah.klein@beispiel.de, ich hatte 3x silent failures, ist da was?\n"
        " Beenden: 'exit' oder Ctrl-C\n"
        "========================================================\n"
    )


def _print_block(label: str, text: str, indent: str = "  ") -> None:
    print(f"{indent}[{label}] {text}")


async def chat() -> None:
    print(_build_greeting())

    # Anthropic-Key dem CLI durchreichen (das SDK liest ANTHROPIC_API_KEY).
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_KEY

    server = create_sdk_mcp_server(
        name="dubly",
        version="0.4.0",
        tools=[
            search_knowledge_base,
            get_customer,
            get_subscription,
            get_credits,
            list_recent_jobs,
            get_job_status,
            create_plain_thread,
            add_plain_note,
            escalate_to_human,
            grant_test_credits,
            restart_lipsync_job,
            apply_credit_bonus,
        ],
    )

    options = ClaudeAgentOptions(
        system_prompt=build_system_prompt(),
        mcp_servers={"dubly": server},
        # Explizit nur unsere Custom-Tools, keine Claude-Code-Defaults:
        tools=[],
        # Read/write Tools sind hier vorher-approved (laufen ohne Prompt).
        # Action-Tools sind ABSICHTLICH NICHT hier -- die gehen durch den
        # approval_callback, der im Terminal nach y/n fragt.
        allowed_tools=[
            "mcp__dubly__search_knowledge_base",
            "mcp__dubly__get_customer",
            "mcp__dubly__get_subscription",
            "mcp__dubly__get_credits",
            "mcp__dubly__list_recent_jobs",
            "mcp__dubly__get_job_status",
            "mcp__dubly__create_plain_thread",
            "mcp__dubly__add_plain_note",
            "mcp__dubly__escalate_to_human",
        ],
        can_use_tool=approval_callback,
        model=CLAUDE_MODEL,
        setting_sources=[],
        max_turns=20,
    )

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_msg = input("\nDu> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBis bald!")
                return
            if not user_msg:
                continue
            if user_msg.lower() in {"exit", "quit", "bye", "tschuess"}:
                print("Bis bald!")
                return

            await client.query(user_msg)

            tool_called = False
            assistant_text_chunks: list[str] = []
            print()

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            tool_called = True
                            args = block.input or {}
                            # Tool-Name kommt als 'mcp__dubly__get_customer' o.ae.
                            short = block.name.split("__")[-1] if "__" in block.name else block.name
                            # Kompakte Argumentdarstellung
                            arg_pairs = ", ".join(
                                f"{k}={str(v)[:40]!r}" for k, v in args.items()
                            )
                            _print_block("Tool", f"{short}({arg_pairs})")
                        elif isinstance(block, ToolResultBlock):
                            # Wird normalerweise innerhalb desselben Stroms gepusht
                            pass
                        elif isinstance(block, TextBlock):
                            assistant_text_chunks.append(block.text)

            answer = "".join(assistant_text_chunks).strip()
            if not tool_called:
                _print_block("Info", "Kein Tool genutzt (Smalltalk/Rueckfrage).", indent="  ")
            print()
            print("Bot>", answer if answer else "(keine Textantwort)")
            if "[ESCALATE]" in answer:
                print("  -> ESKALATION getriggert (in Stufe B legen wir hier einen Plain-Thread an).")


if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nBis bald!")
