"""
Dubly Bot mit RAG (Google Gemini Version)
==========================================
Der Mini-Bot, aber jetzt mit echtem Wissen aus dem Help Center.
Komplett mit Google Gemini Free Tier – kein Anthropic, kein Voyage,
keine Kreditkarte nötig.

Voraussetzung:
    1. scrape_helpcenter.py ausgeführt -> helpcenter_articles.json
    2. embed_chunks.py ausgeführt    -> chunks_with_embeddings.json
    3. pip3 install google-genai numpy

Ausführen:
    python3 dubly_bot_rag.py
"""

from __future__ import annotations

import datetime as dt
import getpass
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import uuid
import webbrowser
import zoneinfo
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    import numpy as np
    import requests
except ImportError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else "google-genai / numpy / requests"
    print(f"Fehlende Bibliothek '{missing}'. Bitte einmal im Terminal:")
    print("    pip3 install google-genai numpy requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
PORT = 8765
MODEL = "gemini-2.0-flash"  # Stabiler als 2.5-flash (das ist oft überlastet)
EMBEDDING_MODEL = "gemini-embedding-001"  # Muss mit embed_chunks.py übereinstimmen
MAX_TOKENS = 1500  # Großzügig, damit Antworten nicht mittendrin abbrechen
RETRY_ON_503 = True  # Bei "model overloaded" automatisch nochmal probieren
TOP_K = 5
MIN_SIMILARITY = 0.55  # Gemini-Embeddings haben tendenziell höhere Scores als Voyage

CHUNKS_FILE = Path(__file__).parent / "chunks_with_embeddings.json"

# Plain (Eskalations-Ziel) – optional. Wenn kein PLAIN_API_KEY gesetzt ist,
# läuft der Bot trotzdem, aber schreibt keine Tickets in plain.
PLAIN_API_URL = "https://core-api.uk.plain.com/graphql/v1"

# ---------------------------------------------------------------------------
# Knowledge Base laden
# ---------------------------------------------------------------------------
if not CHUNKS_FILE.exists():
    print(f"FEHLER: {CHUNKS_FILE.name} nicht gefunden.")
    print("Bitte erst diese zwei Skripte ausführen:")
    print("    python3 scrape_helpcenter.py")
    print("    python3 embed_chunks.py")
    sys.exit(1)

print(f"Lade Wissensbasis aus {CHUNKS_FILE.name}...")
with CHUNKS_FILE.open(encoding="utf-8") as f:
    CHUNKS = json.load(f)

EMBEDDINGS = np.array([c["embedding"] for c in CHUNKS], dtype=np.float32)
EMBEDDINGS = EMBEDDINGS / np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
print(f"  -> {len(CHUNKS)} Chunks bereit (Dimension {EMBEDDINGS.shape[1]})")

# ---------------------------------------------------------------------------
# API-Key
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
PLAIN_API_KEY = os.environ.get("PLAIN_API_KEY")

if not API_KEY:
    print("=" * 60)
    print("Dubly Bot mit RAG (Gemini)")
    print("=" * 60)
    print("Bitte gib deinen Google Gemini API-Key ein.")
    print("(Holen auf aistudio.google.com -> Get API Key)")
    print("-" * 60)
    API_KEY = getpass.getpass("Gemini API-Key: ").strip()
    if not API_KEY:
        print("Kein Key eingegeben. Abbruch.")
        sys.exit(1)

if not PLAIN_API_KEY:
    print()
    print("Plain-Integration (Stufe 2) – optional:")
    print("Wenn du einen Plain API-Key mit Schreibrechten hast, gib ihn ein.")
    print("Dann erstellt der Bot bei Eskalationen echte Tickets in plain.")
    print("Leer lassen + Enter, wenn ohne Plain testen möchtest.")
    PLAIN_API_KEY = getpass.getpass("Plain API-Key (optional): ").strip() or None

client = genai.Client(api_key=API_KEY)

# Session-ID für anonyme Customer-Identifikation in plain
SESSION_ID = str(uuid.uuid4())[:8]
DEMO_CUSTOMER_EXTERNAL_ID = f"bot-demo-{SESSION_ID}"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def embed_query(text: str) -> np.ndarray:
    """Holt das Embedding für eine User-Frage von Gemini."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    emb = np.array(result.embeddings[0].values, dtype=np.float32)
    return emb / np.linalg.norm(emb)


def retrieve_top_chunks(query: str, top_k: int = TOP_K) -> tuple[list[dict], float]:
    """Findet die relevantesten Help-Center-Chunks."""
    query_emb = embed_query(query)
    similarities = EMBEDDINGS @ query_emb
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_chunks = []
    for idx in top_indices:
        chunk = dict(CHUNKS[int(idx)])
        chunk.pop("embedding", None)
        chunk["_score"] = float(similarities[idx])
        top_chunks.append(chunk)
    return top_chunks, float(similarities[top_indices[0]])

# ---------------------------------------------------------------------------
# SLA-Logik (unverändert)
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
        normal_reply = ("'We'll come back to you next business day morning.' "
                        "/ 'Wir melden uns am nächsten Werktag früh.'")
        critical_reply = ("Für kritische Themen: 'I'm flagging this to our on-call team right now – "
                          "you'll hear back as fast as possible.' / 'Das geht jetzt an unsere Notfall-Bereitschaft.'")
    elif is_after_hours:
        situation = "Werktag nach 17 Uhr"
        normal_reply = ("'We'll come back to you tomorrow morning.' "
                        "/ 'Wir melden uns morgen früh bei dir.'")
        critical_reply = normal_reply
    else:
        situation = "Werktag (Mo-Fr) vor 17 Uhr"
        normal_reply = ("'Our team will come back to you with an update today.' "
                        "/ 'Wir melden uns heute noch bei dir.'")
        critical_reply = normal_reply

    return (
        f"AKTUELLE ZEIT-INFO (für SLA-Wahl bei Eskalationen):\n"
        f"- Jetzt: {now.strftime('%A, %d.%m.%Y %H:%M')} (Europe/Berlin)\n"
        f"- Situation: {situation}\n"
        f"- Normale Anfrage → {normal_reply}\n"
        f"- Kritische Anfrage → {critical_reply}"
    )

# ---------------------------------------------------------------------------
# System-Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BASE = """Du bist der KI-Support-Assistent von Dubly.AI – einer SaaS-Plattform für KI-Video-Dubbing.
Du sprichst direkt mit Kunden im Chat.

# DEIN CHARAKTER (Voice Guide)
- Geduzt im Deutschen, "you" im Englischen. Niemals "Sie".
- Warm und kompetent, aber nicht kumpelhaft. "Hi!" als Einstieg ist gut.
- Kurz im Chat: 1–3 Sätze ODER eine präzise Rückfrage.
- Emojis sparsam: max. 1 pro Antwort.
- Sprache spiegeln: DE-Kunde -> DE-Antwort. EN-Kunde -> EN-Antwort.

# WICHTIG: STELL OFT EINE RÜCKFRAGE STATT ZU RATEN

Bevor du eine Antwort gibst, prüfe IMMER:
1. Ist die Frage des Kunden eindeutig? (Welches Video? Welcher Plan? Welche Sprache?)
2. Habe ich genug Kontext, um eine PRÄZISE Antwort zu geben?
3. Könnte meine Antwort wegen fehlender Info am Bedürfnis vorbeigehen?

Wenn Antwort = "nein" zu einem dieser Punkte → stell EINE einzige präzise Rückfrage.
Niemals 3 Fragen auf einmal. Niemals generisch antworten und hoffen, dass es passt.

## Beispiele für gute Rückfragen (genauso machen):

User: "Mein Video lädt nicht hoch."
Bot: "Mist! Damit ich helfen kann: Bekommst du eine Fehlermeldung, oder hängt es einfach bei einem bestimmten Prozentsatz?"

User: "How long does dubbing take?"
Bot: "Depends on your video length and the features you use – are you asking for a quick estimate for a specific video, or in general?"

User: "Kann ich das Video auf Spanisch übersetzen?"
Bot: "Klar! Meinst du Spanisch (Spanien) oder Latein-Spanisch? Wir unterstützen beide separat."

User: "Ich seh meine Credits nicht."
Bot: "Hm – meinst du, der Credit-Counter im Dashboard zeigt 0, oder findest du die Anzeige gar nicht?"

User: "How do I export my dub?"
Bot: "Yes! As MP4 video or as SRT subtitles – which one do you need?"

## Beispiele für schlechte direkte Antworten (VERMEIDEN):

User: "Wie lange dauert das Dubben?"
Schlecht: "Es dauert typischerweise X Minuten..." (Kunde meinte vielleicht ein konkretes Video!)
Besser: "Kommt drauf an – meinst du für ein bestimmtes Video, oder einen Richtwert?"

User: "Mein Video funktioniert nicht."
Schlecht: "Probier mal das Video neu hochzuladen..." (zu vage, vielleicht ein anderes Problem)
Besser: "Was genau geht nicht – Upload, Übersetzung, Wiedergabe?"

## Ausnahme: Wenn die Frage WIRKLICH eindeutig ist, antworte direkt.

User: "Welche Sprachen unterstützt ihr?"
Bot: "Über 30 Sprachen – die volle Liste findest du im Help Center."

User: "Wie funktioniert die Free Trial?"
Bot: "[antworte aus dem Help-Center-Auszug]"

# ZWEI GOLDENE REGELN

## Regel #1: Niemals raten
Wenn die Antwort NICHT eindeutig in den unten gegebenen Help-Center-Auszügen
steht, sage ehrlich "Ich bin mir nicht sicher" und eskaliere an einen Menschen.
NIEMALS plausibel klingende Schritte erfinden.

## Regel #2: Bei Eskalationen verbindlich sein
Drei konkrete Sachen: was du tust, wann der Kunde Antwort bekommt, was er
tun kann falls die Antwort ausbleibt. NIE "as soon as possible".
Die richtige Zeitzusage hängt von der Uhrzeit ab – siehe SLA-Block unten.

# ESKALATIONS-TRIGGER (sofort an Mensch)
- Wörter wie refund, cancel, scammed, kündigen, Beschwerde
- Geldbeträge über 50€/$50
- Emotional aufgeladene Kunden
- Kunde fragt nach Mensch
- Bei zwei Versuchen nicht klar verstanden
- Account-Aktionen (löschen, Plan ändern)
- Spezifische Bug-Reports die Account-Zugriff brauchen

Bei Eskalation am Ende deiner Antwort in eigener Zeile genau das Wort: [ESCALATE]
"""


def build_system_prompt(chunks: list[dict], top_score: float) -> str:
    if top_score >= MIN_SIMILARITY:
        knowledge_block = "\n\n# GEFUNDENE HELP-CENTER-AUSZÜGE\n\n"
        for i, c in enumerate(chunks, start=1):
            knowledge_block += f"## Auszug {i} (Score: {c['_score']:.2f})\n"
            knowledge_block += f"Quelle: {c['url']}\n"
            knowledge_block += f"Titel: {c['title']}\n\n"
            knowledge_block += c["text"] + "\n\n---\n\n"
    else:
        knowledge_block = (
            "\n\n# KEIN PASSENDER HELP-CENTER-AUSZUG GEFUNDEN\n\n"
            f"Der beste Match hatte nur Score {top_score:.2f} (Schwelle: {MIN_SIMILARITY}). "
            "Das heißt: Die Frage wird im Help Center vermutlich NICHT beantwortet. "
            "Sei ehrlich: sage 'Ich bin mir nicht sicher' und eskaliere an einen Menschen."
        )

    sla_block = "\n\n# " + build_sla_context()
    return SYSTEM_PROMPT_BASE + knowledge_block + sla_block


# ---------------------------------------------------------------------------
# Plain-Integration (Eskalations-Ziel)
# ---------------------------------------------------------------------------
PLAIN_UPSERT_CUSTOMER_MUTATION = """
mutation upsertCustomer($input: UpsertCustomerInput!) {
  upsertCustomer(input: $input) {
    result
    customer { id }
    error { message code }
  }
}
"""

PLAIN_CREATE_THREAD_MUTATION = """
mutation createThread($input: CreateThreadInput!) {
  createThread(input: $input) {
    thread { id title }
    error { message code }
  }
}
"""


def plain_request(query: str, variables: dict) -> dict | None:
    """Sendet eine GraphQL-Mutation an plain. Gibt das JSON-Response zurück."""
    if not PLAIN_API_KEY:
        return None
    try:
        resp = requests.post(
            PLAIN_API_URL,
            headers={"Authorization": f"Bearer {PLAIN_API_KEY}", "Content-Type": "application/json"},
            json={"query": query, "variables": variables},
            timeout=15,
        )
    except Exception as e:
        return {"errors": [{"message": f"Verbindungsfehler: {e}"}]}
    if resp.status_code != 200:
        return {"errors": [{"message": f"HTTP {resp.status_code}: {resp.text[:200]}"}]}
    return resp.json()


def plain_upsert_customer(external_id: str, full_name: str) -> tuple[str | None, str | None]:
    """Upsertet einen Demo-Customer in plain. Gibt (customer_id, error) zurück."""
    result = plain_request(PLAIN_UPSERT_CUSTOMER_MUTATION, {
        "input": {
            "identifier": {"externalId": external_id},
            "onCreate": {
                "fullName": full_name,
                "externalId": external_id,
                "email": {"email": f"{external_id}@bot-demo.dubly.ai", "isVerified": False},
            },
            "onUpdate": {},
        }
    })
    if not result:
        return None, "Plain-Request fehlgeschlagen"
    if result.get("errors"):
        return None, str(result["errors"])
    data = (result.get("data") or {}).get("upsertCustomer") or {}
    if data.get("error"):
        return None, str(data["error"])
    return ((data.get("customer") or {}).get("id"), None)


def plain_create_thread(customer_id: str, title: str, body_text: str) -> tuple[str | None, str | None]:
    """Erstellt einen Thread in plain mit dem gegebenen Inhalt als initial message."""
    result = plain_request(PLAIN_CREATE_THREAD_MUTATION, {
        "input": {
            "title": title[:120],
            "customerIdentifier": {"customerId": customer_id},
            "components": [
                {"componentText": {"text": body_text[:9000]}}  # Plain text-length limit
            ],
        }
    })
    if not result:
        return None, "Plain-Request fehlgeschlagen"
    if result.get("errors"):
        return None, str(result["errors"])
    data = (result.get("data") or {}).get("createThread") or {}
    if data.get("error"):
        return None, str(data["error"])
    return ((data.get("thread") or {}).get("id"), None)


def format_chat_transcript(messages: list[dict], escalation_reason: str) -> str:
    """Formatiert den Chat-Verlauf als lesbaren Text für die Plain-Note."""
    lines = [
        "BOT-ESKALATION",
        f"Session-ID: {SESSION_ID}",
        f"Demo-Modus: Diese Konversation kam vom RAG-Bot-Prototyp (Phase-1, lokaler Test).",
        "",
        f"Letzte Bot-Antwort (vor Eskalation):",
        f"  {escalation_reason[:300]}",
        "",
        "Chat-Verlauf:",
        "-" * 40,
    ]
    for m in messages:
        role_label = "KUNDE" if m["role"] == "user" else "BOT"
        content = m["content"].replace("[ESCALATE]", "").strip()
        lines.append(f"\n[{role_label}]")
        lines.append(content)
    lines.append("-" * 40)
    return "\n".join(lines)


def handoff_to_plain(messages: list[dict]) -> dict:
    """Komplettes Hand-off an plain: Customer upserten + Thread anlegen."""
    if not PLAIN_API_KEY:
        return {"skipped": True, "reason": "Kein Plain-Key konfiguriert"}

    first_user = next((m["content"] for m in messages if m["role"] == "user"), "Bot-Eskalation")
    title = f"[BOT-DEMO] {first_user[:80]}"

    customer_id, err = plain_upsert_customer(
        DEMO_CUSTOMER_EXTERNAL_ID,
        f"Bot Demo Session {SESSION_ID}",
    )
    if not customer_id:
        return {"error": f"Customer-Upsert fehlgeschlagen: {err}"}

    last_bot = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    reason = last_bot.replace("[ESCALATE]", "").strip()
    transcript = format_chat_transcript(messages, reason)

    thread_id, err = plain_create_thread(customer_id, title, transcript)
    if not thread_id:
        return {"error": f"Thread-Creation fehlgeschlagen: {err}"}

    return {"thread_id": thread_id, "title": title, "session": SESSION_ID}


# ---------------------------------------------------------------------------
# Gemini API helpers
# ---------------------------------------------------------------------------
def anthropic_to_gemini_messages(messages: list[dict]) -> list[dict]:
    """Konvertiert Anthropic-style (role: user/assistant) zu Gemini (user/model)."""
    out = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        out.append({"role": role, "parts": [{"text": m["content"]}]})
    return out

# ---------------------------------------------------------------------------
# HTML-Page
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dubly Bot mit RAG</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    background: #f4f6fa; color: #1a1a1a;
    min-height: 100vh; display: flex; flex-direction: column;
  }
  header {
    background: #1F3A5F; color: white;
    padding: 14px 20px; display: flex; align-items: center; gap: 12px;
  }
  header .logo {
    width: 36px; height: 36px; background: white; color: #1F3A5F;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 18px;
  }
  header .title { font-weight: 600; font-size: 16px; }
  header .subtitle { font-size: 12px; opacity: 0.7; }
  header .badge {
    margin-left: auto; background: #10b981; color: white;
    padding: 4px 10px; border-radius: 12px; font-size: 11px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }
  #chat { flex: 1; max-width: 720px; width: 100%; margin: 0 auto; padding: 20px; overflow-y: auto; }
  .msg { margin-bottom: 16px; display: flex; gap: 10px; }
  .msg.user { flex-direction: row-reverse; }
  .msg .avatar {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: bold;
  }
  .msg.bot .avatar { background: #1F3A5F; color: white; }
  .msg.user .avatar { background: #e1e5eb; color: #555; }
  .bubble {
    padding: 10px 14px; border-radius: 14px; max-width: 80%;
    line-height: 1.5; font-size: 14px;
    white-space: pre-wrap; word-wrap: break-word;
  }
  .msg.bot .bubble { background: white; border: 1px solid #e1e5eb; border-bottom-left-radius: 4px; }
  .msg.user .bubble { background: #1F3A5F; color: white; border-bottom-right-radius: 4px; }
  .msg.escalated .bubble { background: #fff8e1; border-color: #f59f00; }
  .escalation-note { font-size: 11px; color: #b45309; margin-top: 6px; font-style: italic; }
  .handoff-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
  }
  .handoff-badge.success { background: #d1fae5; color: #065f46; border: 1px solid #10b981; }
  .handoff-badge.error   { background: #fee2e2; color: #7f1d1d; border: 1px solid #ef4444; }
  .handoff-badge.skipped { background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; }
  .sources { font-size: 11px; color: #666; margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; }
  .sources a { color: #1F3A5F; text-decoration: none; }
  .sources a:hover { text-decoration: underline; }
  .typing { display: inline-flex; gap: 4px; padding: 6px 0; }
  .typing span {
    width: 6px; height: 6px; background: #999; border-radius: 50%;
    animation: blink 1.4s infinite both;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }
  #input-area { background: white; border-top: 1px solid #e1e5eb; padding: 14px 20px; }
  #input-form { max-width: 720px; margin: 0 auto; display: flex; gap: 10px; }
  #input {
    flex: 1; border: 1px solid #d1d5db; border-radius: 22px;
    padding: 10px 16px; font-size: 14px; font-family: inherit;
    resize: none; max-height: 120px; outline: none;
  }
  #input:focus { border-color: #1F3A5F; }
  #send {
    background: #1F3A5F; color: white; border: none;
    border-radius: 50%; width: 40px; height: 40px;
    cursor: pointer; font-size: 18px;
    display: flex; align-items: center; justify-content: center;
  }
  #send:disabled { opacity: 0.4; cursor: not-allowed; }
  #send:hover:not(:disabled) { background: #2a4d7a; }
  .hint { text-align: center; color: #888; font-size: 12px; padding: 6px 20px 12px; }
  .examples {
    max-width: 720px; margin: 0 auto 16px;
    display: flex; flex-wrap: wrap; gap: 8px; padding: 0 20px;
  }
  .example {
    background: white; border: 1px solid #e1e5eb;
    border-radius: 14px; padding: 8px 12px;
    font-size: 12px; cursor: pointer; color: #555;
  }
  .example:hover { background: #f0f4fa; border-color: #1F3A5F; color: #1F3A5F; }
</style>
</head>
<body>
<header>
  <div class="logo">D</div>
  <div>
    <div class="title">Dubly Support-Bot</div>
    <div class="subtitle">Mit echtem Help-Center-Wissen · Gemini</div>
  </div>
  <div class="badge">RAG aktiv</div>
</header>
<div id="chat">
  <div class="msg bot">
    <div class="avatar">D</div>
    <div class="bubble">Hi 👋 Ich bin der Dubly-Support-Bot. Diesmal mit echtem Wissen aus dem gesamten Help Center. Stell mir gerne Fragen zum Produkt – ich antworte auf Deutsch oder Englisch.</div>
  </div>
</div>
<div class="examples" id="examples">
  <div class="example">Kann ich nebeneinander mehrere Videos hochladen?</div>
  <div class="example">How do I export a dub as SRT?</div>
  <div class="example">Wie funktioniert die Free Trial?</div>
  <div class="example">Do you support Arabic lip sync?</div>
  <div class="example">I want a refund for my subscription!</div>
</div>
<div id="input-area">
  <form id="input-form">
    <textarea id="input" rows="1" placeholder="Schreib deine Nachricht..." autofocus></textarea>
    <button id="send" type="submit">↑</button>
  </form>
  <div class="hint">Phase-1 Demo · Antworten basieren auf 79 Help-Center-Artikeln</div>
</div>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('input-form');
const input = document.getElementById('input');
const send = document.getElementById('send');
const examples = document.getElementById('examples');
const history = [];

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
examples.addEventListener('click', (e) => {
  if (e.target.classList.contains('example')) {
    input.value = e.target.textContent;
    input.focus();
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  examples.style.display = 'none';
  appendMessage('user', text);
  history.push({role: 'user', content: text});
  input.value = '';
  input.style.height = 'auto';
  send.disabled = true;
  const typingEl = appendTyping();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages: history})
    });
    const data = await res.json();
    typingEl.remove();
    if (data.error) {
      appendMessage('bot', '⚠️ Fehler: ' + data.error);
    } else {
      let reply = data.reply || '';
      let escalated = false;
      if (reply.includes('[ESCALATE]')) {
        escalated = true;
        reply = reply.replace(/\\[ESCALATE\\]/g, '').trim();
      }
      appendMessage('bot', reply, escalated, data.sources || [], data.handoff);
      history.push({role: 'assistant', content: data.reply});
    }
  } catch (err) {
    typingEl.remove();
    appendMessage('bot', '⚠️ Verbindungsfehler: ' + err.message);
  }
  send.disabled = false;
  input.focus();
});

function appendMessage(role, text, escalated = false, sources = [], handoff = null) {
  const msg = document.createElement('div');
  msg.className = 'msg ' + role + (escalated ? ' escalated' : '');
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'bot' ? 'D' : 'Du';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  if (escalated) {
    const note = document.createElement('div');
    note.className = 'escalation-note';
    note.textContent = '↗ Bot hat an einen Menschen übergeben';
    bubble.appendChild(note);
  }
  if (handoff) {
    const badge = document.createElement('div');
    if (handoff.thread_id) {
      badge.className = 'handoff-badge success';
      badge.textContent = '✓ Plain-Ticket erstellt: ' + handoff.thread_id;
    } else if (handoff.error) {
      badge.className = 'handoff-badge error';
      badge.textContent = '⚠ Plain-Übergabe fehlgeschlagen: ' + handoff.error.substring(0, 100);
    } else if (handoff.skipped) {
      badge.className = 'handoff-badge skipped';
      badge.textContent = 'ℹ Plain nicht konfiguriert (kein Ticket erstellt)';
    }
    bubble.appendChild(badge);
  }
  if (sources && sources.length) {
    const srcDiv = document.createElement('div');
    srcDiv.className = 'sources';
    srcDiv.innerHTML = 'Quellen: ' + sources.map(s =>
      `<a href="${s.url}" target="_blank">${s.title}</a>`
    ).join(' · ');
    bubble.appendChild(srcDiv);
  }
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

function appendTyping() {
  const msg = document.createElement('div');
  msg.className = 'msg bot';
  msg.innerHTML = '<div class="avatar">D</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP-Handler
# ---------------------------------------------------------------------------
class BotHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages", [])
        except Exception as e:
            self._json(400, {"error": f"Bad request: {e}"}); return

        try:
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            top_chunks, top_score = retrieve_top_chunks(last_user_msg)
            system_prompt = build_system_prompt(top_chunks, top_score)

            gemini_messages = anthropic_to_gemini_messages(messages)

            # Generate-Call mit Retry bei 503 (Modell überlastet)
            response = None
            last_error = None
            for attempt in range(3 if RETRY_ON_503 else 1):
                try:
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=MAX_TOKENS,
                            temperature=0.7,
                        ),
                    )
                    break
                except Exception as gen_err:
                    last_error = gen_err
                    if "503" in str(gen_err) or "UNAVAILABLE" in str(gen_err):
                        wait = (attempt + 1) * 2  # 2s, 4s, 6s
                        print(f"  Modell überlastet (503), warte {wait}s und probiere nochmal...")
                        time.sleep(wait)
                        continue
                    raise
            if response is None:
                raise last_error or RuntimeError("Generate fehlgeschlagen")

            # Robuste Text-Extraktion: erst response.text versuchen,
            # falls leer dann alle Text-Parts aus den Candidates sammeln
            text = ""
            try:
                text = response.text or ""
            except Exception:
                text = ""
            if not text and response.candidates:
                cand = response.candidates[0]
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        if hasattr(part, "text") and part.text:
                            text += part.text

            # Debug-Info im Terminal-Log: finish_reason mitloggen,
            # damit wir sehen falls etwas abgeschnitten wird
            try:
                finish = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                if str(finish) not in ("STOP", "FinishReason.STOP", "1"):
                    print(f"  [WARN] Ungewöhnlicher finish_reason: {finish} | Text-Länge: {len(text)}")
            except Exception:
                pass

            sources = []
            if top_score >= MIN_SIMILARITY:
                seen_urls = set()
                for c in top_chunks[:3]:
                    if c["url"] not in seen_urls:
                        seen_urls.add(c["url"])
                        sources.append({"url": c["url"], "title": c["title"]})

            # Hand-off an plain, falls Bot eskaliert hat
            handoff_info = None
            if "[ESCALATE]" in text:
                # Inklusive der gerade generierten Bot-Antwort
                full_messages = messages + [{"role": "assistant", "content": text}]
                handoff_info = handoff_to_plain(full_messages)
                if "thread_id" in handoff_info:
                    print(f"  [PLAIN] Thread erstellt: {handoff_info['thread_id']}")
                elif "error" in handoff_info:
                    print(f"  [PLAIN] FEHLER: {handoff_info['error'][:150]}")

            self._json(200, {
                "reply": text,
                "sources": sources,
                "rag_score": round(top_score, 3),
                "handoff": handoff_info,
            })
        except Exception as e:
            err_msg = str(e)
            if "quota" in err_msg.lower() or "429" in err_msg:
                err_msg = "Rate Limit erreicht (15 Anfragen/min im Free Tier). Kurz warten und nochmal probieren."
            elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                err_msg = "Modell gerade überlastet bei Google. Kurz warten und nochmal probieren – oder MODEL im Skript auf 'gemini-2.0-flash-lite' setzen."
            self._json(500, {"error": err_msg})

    def _json(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        return


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def open_browser_delayed(url: str) -> None:
    time.sleep(1.2)
    webbrowser.open(url)


def main() -> None:
    url = f"http://localhost:{PORT}"
    print()
    print("=" * 60)
    print(f"  Dubly Bot mit RAG (Gemini) läuft auf {url}")
    print(f"  Modell:        {MODEL}")
    print(f"  Wissensbasis:  {len(CHUNKS)} Chunks aus dem Help Center")
    print(f"  Plain:         {'AKTIV (Tickets werden erstellt)' if PLAIN_API_KEY else 'nicht konfiguriert'}")
    if PLAIN_API_KEY:
        print(f"  Session-ID:    {SESSION_ID} (Customer-Identifier in plain)")
    print("  Browser öffnet sich gleich automatisch.")
    print("  Zum Beenden:   Ctrl + C im Terminal drücken.")
    print("=" * 60)
    print()
    threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()
    server = ThreadingServer(("localhost", PORT), BotHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nBot beendet. Bis bald!\n")
        server.server_close()


if __name__ == "__main__":
    main()
