"""
Anthropic-API-Key Diagnose
==========================
Schickt eine winzige Test-Anfrage an Claude, um zu prüfen, ob dein
Key valide ist. Zeigt klare Fehler bei den häufigsten Problemen.

Ausführen:
    python3 test_claude_key.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path


def load_env(path: Path) -> None:
    """Mini-.env-Parser (keine externe Lib nötig)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env(Path(__file__).parent / ".env")

try:
    import anthropic
except ImportError:
    print("[FEHLER] Bibliothek 'anthropic' fehlt.")
    print("        Bitte einmal im Terminal:")
    print("            pip3 install anthropic")
    sys.exit(1)


api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

if not api_key:
    print("[FEHLER] Kein ANTHROPIC_API_KEY gefunden.")
    print("        Trag deinen Key in die Datei .env ein (gleicher Ordner).")
    print("        Beispiel-Zeile:  ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(2)

if not api_key.startswith("sk-ant-"):
    print("[WARNUNG] Der Key sieht nicht aus wie ein echter Anthropic-Key")
    print("          (sollte mit 'sk-ant-' beginnen).")
    print("          Trotzdem versuchen ...")
    print()

client = anthropic.Anthropic(api_key=api_key)

print("Sende Test-Anfrage an Claude (Sonnet 4.5) ...")
try:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": "Antworte exakt mit dem Wort 'pong' und sonst nichts.",
            }
        ],
    )
except anthropic.AuthenticationError:
    print("[FEHLER] Authentifizierung abgelehnt.")
    print("        Der Key ist falsch, abgelaufen oder fuer das Konto deaktiviert.")
    print("        Auf console.anthropic.com -> API Keys neu erstellen.")
    sys.exit(3)
except anthropic.RateLimitError:
    print("[FEHLER] Rate-Limit erreicht. Eine Minute warten und neu probieren.")
    sys.exit(4)
except anthropic.APIStatusError as e:
    print(f"[FEHLER] API-Fehler {e.status_code}: {e.message}")
    sys.exit(5)
except Exception as e:  # noqa: BLE001
    print(f"[FEHLER] Unerwartet: {type(e).__name__}: {e}")
    sys.exit(6)

text = "".join(block.text for block in msg.content if hasattr(block, "text"))
in_tok = msg.usage.input_tokens if msg.usage else "?"
out_tok = msg.usage.output_tokens if msg.usage else "?"

print()
print("------------------------------------------------------------")
print("  Antwort von Claude:", text.strip())
print(f"  Token-Verbrauch:    {in_tok} input, {out_tok} output")
print(f"  Kosten dieses Pings: ca. 0,001 EUR")
print("------------------------------------------------------------")
print()
print("ALLES OK. Dein Key funktioniert.")
print("Naechster Schritt:  python3 dubly_agent.py")
