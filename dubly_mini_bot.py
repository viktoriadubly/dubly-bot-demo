"""
Dubly Mini-Bot – Phase-0 Prototyp
==================================
Ein einfacher Chatbot, mit dem du die Bot-Stimme und Conversational-Qualität
in echt ausprobieren kannst, bevor wir den richtigen Bot bauen.

Was er kann:
  - Pre-Sales-Fragen zum Dubly Help Center beantworten
  - In DE und EN unterhalten
  - Bei sensiblen Themen (Refund, etc.) korrekt eskalieren
  - Den Voice Guide befolgen (warm, geduzt, kurz im Chat)

Wie du ihn startest (im Terminal):
    pip3 install anthropic
    python3 dubly_mini_bot.py

Dann öffnet sich automatisch ein Browser-Fenster mit dem Chat.
Zum Beenden: Ctrl + C im Terminal.
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
import webbrowser
import zoneinfo

# ---------------------------------------------------------------------------
# Abhängigkeit prüfen
# ---------------------------------------------------------------------------
try:
    from anthropic import Anthropic
except ImportError:
    print("Fehlende Bibliothek 'anthropic'. Bitte einmal im Terminal:")
    print("    pip3 install anthropic")
    print("...und dann das Skript erneut starten.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
PORT = 8765
MODEL = "claude-sonnet-4-5"   # Falls dein Account das noch nicht hat: claude-haiku-4-5 ist günstiger
MAX_TOKENS = 700

# ---------------------------------------------------------------------------
# API-Key holen
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("=" * 60)
    print("Dubly Mini-Bot")
    print("=" * 60)
    print("Bitte gib deinen Anthropic API-Key ein.")
    print("(Beginnt mit 'sk-ant-...'. Während du tippst,")
    print(" siehst du nichts auf dem Bildschirm.)")
    print("-" * 60)
    API_KEY = getpass.getpass("API-Key: ").strip()
    if not API_KEY:
        print("Kein Key eingegeben. Abbruch.")
        sys.exit(1)

client = Anthropic(api_key=API_KEY)

# ---------------------------------------------------------------------------
# Wissensbasis (extrahiert aus eurem Help Center support.dubly.ai)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = """
# Dubly.AI – Produkt-Übersicht

Dubly.AI ist eine SaaS-Plattform für KI-gestütztes Video-Dubbing.
Kernfunktionen: Übersetzung von Videos in andere Sprachen, AI-Voiceover
(Studio Voices + Custom Voice Clone), Lip-Sync, SRT-Untertitel-Export,
Translation-Editor mit Glossar und Brand-Terminologie. Modell: Subscription
+ Credits, mit Free Trial.

Zielgruppe: Content Creator, Marketing-/Video-Teams, Workspaces mit
mehreren Mitgliedern (Reviewer-Workflow).

# Help-Center-Struktur (9 Kategorien, 79 Artikel)

## 1. Getting Started
- What is Dubly.AI?
- Create an account and log in
- Quick-start guide: Your first dub
- Dashboard overview: Where to find everything
- Key dubbing terms explained
- Contact support
- Free trial: Translate your first video at no cost

## 2. Users, Roles & Permissions
- Transfer ownership of a Workspace
- Remove a member from your account
- Off-board a member
- What happens to a member's content when their account is deleted
- Invite reviewers to check translations
- Invite team members and manage roles
- Organizations and user roles

## 3. Voice & Lip Sync
- When to turn on lip-sync
- Filming tips for perfect lip-sync
- Handle multiple speakers on camera
- What is Custom Voice Clone
- Preserve emotion and tone across languages
- Lip-sync limitations and workarounds

## 4. Billing & Subscription
- Pricing Model and Translation Costs
- Billing and pricing overview
- Change or upgrade your plan
- Cancellation policy
- Understand your invoice
- Fix failed payments and billing errors
- Refunds and credit recovery
- Pay for credits on demand
- Why a plan auto-renews
- Information about credit usage
- Export usage data
- Discounts
- Pause a subscription
- When credits reset

## 5. Translate a Video
- Upload a video: supported formats
- Choose source and target languages
- Voice options: Studio voices and voice cloning
- Preview, export and download your dub
- Organize work: Projects, folders, versions
- Fine-tune with the Advanced Editor
- How long dubbing takes
- What happens when you delete a video
- Export dub as SRT subtitle file
- Re-translate after editing the source video
- Share a dub
- Lip-sync basics
- Review and edit translations
- Request a custom batch project
- Edit a segment: original text and translation

## 6. Get the Best Results
- Do's and don'ts for source videos
- Translation Styles & Glossary
- Music & Sound Effects Best Practices
- Improve translation quality
- Change translation style on already dubbed video

## 7. Troubleshoot
- Fix a failed upload
- Fix errors in translation
- Fix download and export issues
- Fix robotic or unnatural-sounding voice
- Video is stuck
- Browser and network issues
- Error with video
- Processing takes too long
- Input error
- Lip-sync error
- Voiceover sounds inconsistent or distorted
- Can't access account
- Reset password link not received
- Fix common lip-sync artifacts

## 8. Privacy & Security
- Acceptable use policy
- Who owns uploaded and dubbed content
- Data retention and deletion periods
- EU AI Act: What you need to know
- How to disclose AI dubbing to viewers
- Upload and monetize content
- Data security and privacy at Dubly.AI

## 9. Account Management
- Edit profile
- Delete account
- Multiple Accounts/Workspaces
- Reset password

# Verlinkungs-Schema
Artikel-URLs folgen dem Muster: https://support.dubly.ai/articles/<slug>
Beispiele: /articles/quick-start-guide-your-first-dub, /articles/what-is-dubly-ai
"""

# ---------------------------------------------------------------------------
# SLA-Logik (basierend auf Voice Guide v2)
# ---------------------------------------------------------------------------
# Dubly arbeitet werktags 9-17 Uhr (Europe/Berlin). Wochenende = Bereitschaft
# nur für kritische Themen. Diese Funktion berechnet die richtige Zeitzusage
# je nach Uhrzeit/Wochentag.
DUBLY_TZ = zoneinfo.ZoneInfo("Europe/Berlin")
WORKDAY_END_HOUR = 17  # nach 17 Uhr -> "morgen früh"


def build_sla_context() -> str:
    """Liefert einen kurzen Hinweis zur aktuell gültigen SLA-Stufe."""
    now = dt.datetime.now(DUBLY_TZ)
    weekday = now.weekday()  # Mo=0 ... So=6
    is_weekend = weekday >= 5
    is_after_hours = now.hour >= WORKDAY_END_HOUR

    if is_weekend:
        situation = "Wochenende"
        normal_reply = ("Für normale Anfragen: 'We'll come back to you next business day morning.' "
                        "/ 'Wir melden uns am nächsten Werktag früh.'")
        critical_reply = ("Für kritische Themen (Refund, Beschwerde, akuter Schaden): "
                          "'I'm flagging this to our on-call team right now – you'll hear back as fast as possible.' "
                          "/ 'Das geht jetzt an unsere Notfall-Bereitschaft, du hörst so schnell wie möglich von uns.'")
    elif is_after_hours:
        situation = "Werktag nach 17 Uhr"
        normal_reply = ("'We'll come back to you tomorrow morning.' "
                        "/ 'Wir melden uns morgen früh bei dir.'")
        critical_reply = normal_reply + " (Bei kritischen Themen ggf. Bereitschaft signalisieren.)"
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
        f"- Kritische Anfrage → {critical_reply}\n"
        f"WICHTIG: Niemals 'today' versprechen, wenn die Eingangszeit das nicht zulässt – das wäre eine Lüge."
    )


# ---------------------------------------------------------------------------
# System-Prompt (basiert auf dem Voice Guide v2 aus Schritt 3)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_BASE = f"""Du bist der KI-Support-Assistent von Dubly.AI – einer SaaS-Plattform für KI-Video-Dubbing.
Du sprichst direkt mit Kunden im Chat oder per E-Mail.

# DEIN CHARAKTER (Voice Guide)

- Geduzt im Deutschen, "you" im Englischen. Niemals "Sie" verwenden.
- Warm und kompetent, aber nicht kumpelhaft. "Hi!" als Einstieg ist gut.
- Kurz im Chat: 1–3 Sätze. Nur länger antworten, wenn der Kontext (E-Mail, komplexer Sachverhalt) es verlangt.
- Emojis sparsam: max. 1 pro Antwort. Geeignet: 👋 zur Begrüßung, ✅ bei erledigt. KEIN 🚀/😂/🎉.
- Sprache spiegeln: Schreibt der Kunde Deutsch, antwortest du Deutsch. Englisch zu Englisch. Niemals mischen.
- Bei Unsicherheit ehrlich sein: niemals raten oder Sachen erfinden.

# ZWEI GOLDENE REGELN (nicht verhandelbar)

## Regel #1: Niemals raten oder erfinden
Wenn die Antwort nicht eindeutig in der Wissensbasis steht, sage ehrlich
"Ich weiß es nicht / ich will nichts erfinden" und eskaliere an einen Menschen.
Niemals plausibel klingende Schritte vorschlagen, die du nicht aus der Quelle hast.

Falsches Beispiel: "Geh in den Editor, lösche das erste Segment..." (wenn du nicht
sicher weißt, dass das funktioniert).
Richtiges Beispiel: "Bin mir bei der genauen Lösung nicht sicher – damit ich dir
nichts Falsches sage, ziehe ich jemanden aus dem Team dran."

## Regel #2: Verbindlich eskalieren, nicht vage
Bei Eskalationen IMMER drei konkrete Sachen: (1) was du tust, (2) wann der Kunde
antwortet bekommt, (3) was er tun kann falls die Antwort nicht kommt.
NIE Floskeln wie "as soon as possible" oder "we'll get back to you".

Die richtige Zeitzusage hängt von der aktuellen Uhrzeit ab – siehe SLA-Block unten.

# DEIN WISSEN

Du kennst das komplette Dubly Help Center (siehe unten). Wenn ein Kunde eine
Pre-Sales- oder How-to-Frage stellt, beantworte sie auf Basis dieses Wissens.

Wenn die Antwort im Help Center steht, fasse sie in eigenen Worten zusammen und
verweise auf den passenden Artikel (z. B. "Mehr dazu in unserem Help-Center-
Artikel 'Lip-sync basics'.").

# WANN DU UMGEHEND AN EINEN MENSCHEN ÜBERGIBST (Eskalation)

- Kunde nennt Wörter wie: refund, scammed, cancel my subscription, kündigen,
  rechtliche Schritte, Beschwerde, "I want a refund", "money back"
- Geldbeträge über 50€/$50 werden erwähnt
- Kunde ist emotional aufgeladen (mehrere Ausrufezeichen, Großbuchstaben,
  harte Wörter, Frust deutlich spürbar)
- Kunde fragt explizit nach einem Menschen ("Ich will mit jemandem sprechen")
- Du bist dir bei zwei Versuchen nicht sicher / hast den Kunden missverstanden
- Account-Aktionen (Account löschen, Plan ändern, Daten exportieren)
- Konkretes Account-Problem (z. B. "meine Credits sind weg") – du kannst
  Account-Daten nicht sehen, also weiter an Mensch

Bei Eskalation sagst du sinngemäß (in der Sprache des Kunden):
"Verstehe – das geht jemand aus dem Team direkt an. Ich übergebe das Gespräch
jetzt mit allem Kontext, damit du nichts nochmal erklären musst. Du hörst
gleich von uns."

Dann markiere intern (am Ende deiner Nachricht in einer eigenen Zeile):
[ESCALATE]

Das [ESCALATE]-Tag ist nur für interne Erkennung – der Kunde sieht es nicht.

# DEINE ERSTE NACHRICHT

Wenn das die erste Nachricht im Gespräch ist (User-Message ist die einzige),
begrüße kurz und frag nach dem Anliegen. Sonst: direkt auf die Frage eingehen.

# WENN DU NACHFRAGEN MUSST

Wenn die Frage des Kunden mehrdeutig ist oder dir Infos fehlen (z. B. Welches
Video? Welcher Plan? Welcher Browser?), stelle GENAU EINE präzise Rückfrage.
Niemals 3 Fragen auf einmal.

# WISSENSBASIS (Help Center)

{KNOWLEDGE_BASE}
"""


def build_system_prompt() -> str:
    """Setzt den finalen System-Prompt mit aktuellem SLA-Context zusammen."""
    return SYSTEM_PROMPT_BASE + "\n\n# " + build_sla_context()

# ---------------------------------------------------------------------------
# Embedded Chat-UI (HTML/CSS/JS in einer Datei)
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dubly Mini-Bot – Prototyp</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    background: #f4f6fa;
    color: #1a1a1a;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: #1F3A5F;
    color: white;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
  }
  header .logo {
    width: 36px; height: 36px;
    background: white; color: #1F3A5F;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 18px;
  }
  header .title { font-weight: 600; font-size: 16px; }
  header .subtitle { font-size: 12px; opacity: 0.7; margin-top: 2px; }
  header .badge {
    margin-left: auto;
    background: #f59f00; color: white;
    padding: 4px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  #chat {
    flex: 1;
    max-width: 720px;
    width: 100%;
    margin: 0 auto;
    padding: 20px;
    overflow-y: auto;
  }
  .msg {
    margin-bottom: 16px;
    display: flex;
    gap: 10px;
  }
  .msg.user { flex-direction: row-reverse; }
  .msg .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: bold;
  }
  .msg.bot .avatar { background: #1F3A5F; color: white; }
  .msg.user .avatar { background: #e1e5eb; color: #555; }
  .bubble {
    padding: 10px 14px;
    border-radius: 14px;
    max-width: 80%;
    line-height: 1.5;
    font-size: 14px;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.bot .bubble {
    background: white;
    border: 1px solid #e1e5eb;
    border-bottom-left-radius: 4px;
  }
  .msg.user .bubble {
    background: #1F3A5F;
    color: white;
    border-bottom-right-radius: 4px;
  }
  .msg.escalated .bubble {
    background: #fff8e1;
    border-color: #f59f00;
  }
  .escalation-note {
    font-size: 11px;
    color: #b45309;
    margin-top: 6px;
    font-style: italic;
  }
  .typing {
    display: inline-flex; gap: 4px;
    padding: 6px 0;
  }
  .typing span {
    width: 6px; height: 6px;
    background: #999;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink {
    0%, 80%, 100% { opacity: 0.3; }
    40% { opacity: 1; }
  }
  #input-area {
    background: white;
    border-top: 1px solid #e1e5eb;
    padding: 14px 20px;
  }
  #input-form {
    max-width: 720px;
    margin: 0 auto;
    display: flex;
    gap: 10px;
  }
  #input {
    flex: 1;
    border: 1px solid #d1d5db;
    border-radius: 22px;
    padding: 10px 16px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    max-height: 120px;
    outline: none;
  }
  #input:focus { border-color: #1F3A5F; }
  #send {
    background: #1F3A5F;
    color: white;
    border: none;
    border-radius: 50%;
    width: 40px; height: 40px;
    cursor: pointer;
    font-size: 18px;
    display: flex; align-items: center; justify-content: center;
  }
  #send:disabled { opacity: 0.4; cursor: not-allowed; }
  #send:hover:not(:disabled) { background: #2a4d7a; }
  .hint {
    text-align: center;
    color: #888;
    font-size: 12px;
    padding: 6px 20px 12px;
  }
  .examples {
    max-width: 720px;
    margin: 0 auto 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 0 20px;
  }
  .example {
    background: white;
    border: 1px solid #e1e5eb;
    border-radius: 14px;
    padding: 8px 12px;
    font-size: 12px;
    cursor: pointer;
    color: #555;
  }
  .example:hover { background: #f0f4fa; border-color: #1F3A5F; color: #1F3A5F; }
</style>
</head>
<body>
<header>
  <div class="logo">D</div>
  <div>
    <div class="title">Dubly Support-Bot</div>
    <div class="subtitle">Prototyp – nur lokal auf deinem Mac</div>
  </div>
  <div class="badge">Beta</div>
</header>
<div id="chat">
  <div class="msg bot">
    <div class="avatar">D</div>
    <div class="bubble">Hi 👋 Ich bin ein erster Test-Bot für Dubly. Stell mir gerne Fragen zum Produkt, dem Übersetzungs-Workflow, Lip-Sync, deinem Account – worüber du willst. Ich antworte auf Deutsch oder Englisch, je nachdem wie du schreibst.</div>
  </div>
</div>
<div class="examples" id="examples">
  <div class="example">Kann ich nebeneinander mehrere Videos hochladen?</div>
  <div class="example">Do you support Arabic lip sync?</div>
  <div class="example">Wie funktioniert die Free Trial?</div>
  <div class="example">I want a refund for my subscription!</div>
  <div class="example">Mein Video hängt seit Stunden</div>
</div>
<div id="input-area">
  <form id="input-form">
    <textarea id="input" rows="1" placeholder="Schreib deine Nachricht... (Enter = senden, Shift+Enter = neue Zeile)" autofocus></textarea>
    <button id="send" type="submit">↑</button>
  </form>
  <div class="hint">Phase-0 Prototyp · Antworten basieren auf eurem Help Center</div>
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
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
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
      appendMessage('bot', reply, escalated);
      history.push({role: 'assistant', content: data.reply});
    }
  } catch (err) {
    typingEl.remove();
    appendMessage('bot', '⚠️ Verbindungsfehler: ' + err.message);
  }
  send.disabled = false;
  input.focus();
});

function appendMessage(role, text, escalated = false) {
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
    note.textContent = '↗ Bot würde jetzt an einen Menschen übergeben';
    bubble.appendChild(note);
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
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages", [])
        except Exception as e:
            self._json(400, {"error": f"Bad request: {e}"})
            return

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=build_system_prompt(),
                messages=messages,
            )
            text = response.content[0].text
            self._json(200, {"reply": text})
        except Exception as e:
            err_msg = str(e)
            if "model" in err_msg.lower() or "not_found" in err_msg.lower():
                err_msg += "\n\nTipp: Im Skript-Header MODEL auf 'claude-haiku-4-5' setzen."
            self._json(500, {"error": err_msg})

    def _json(self, status, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        # Stille Logs (sonst spammt's das Terminal)
        return


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def open_browser_delayed(url: str) -> None:
    time.sleep(1.2)
    webbrowser.open(url)


def main() -> None:
    url = f"http://localhost:{PORT}"
    print()
    print("=" * 60)
    print(f"  Dubly Mini-Bot läuft auf {url}")
    print(f"  Modell:        {MODEL}")
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
