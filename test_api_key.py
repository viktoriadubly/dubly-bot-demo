"""
Plain API-Key Test
==================
Prüft in wenigen Sekunden, ob dein Plain API-Key funktioniert und
welche Berechtigungen er hat. Lädt KEINE Tickets herunter.

Ausführen mit:
    python3 test_api_key.py
"""

from __future__ import annotations

import getpass
import json
import sys

try:
    import requests
except ImportError:
    print("Fehlende Bibliothek 'requests'. Bitte einmal im Terminal ausführen:")
    print("    pip3 install requests")
    sys.exit(1)

PLAIN_API_URL = "https://core-api.uk.plain.com/graphql/v1"

print("=" * 60)
print("Plain API-Key Test")
print("=" * 60)
print("Gib deinen API-Key ein (er wird nicht sichtbar angezeigt):")
api_key = getpass.getpass("API-Key: ").strip()

if not api_key:
    print("Kein Key eingegeben.")
    sys.exit(1)

print()
print(f"Key beginnt mit: {api_key[:14]}...")
print(f"Key-Länge: {len(api_key)} Zeichen")
print()

# Simple Test-Query: einen einzigen Thread anfragen (testet Auth + Permission auf einmal)
test_query = """
query TestKey {
  threads(first: 1) {
    edges { node { id title } }
  }
}
"""

print("Verbinde mich mit Plain...")
try:
    response = requests.post(
        PLAIN_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": test_query},
        timeout=15,
    )
except Exception as e:
    print(f"Verbindungsfehler: {e}")
    sys.exit(1)

print(f"Antwort-Status: HTTP {response.status_code}")
print()

if response.status_code == 401:
    print("FEHLER: Plain sagt 'Unauthorized' (Status 401).")
    print()
    print("Das bedeutet eins von dreien:")
    print("  1. Der Key ist falsch/abgetippt (häufig: Leerzeichen vorne dran).")
    print("  2. Der Key wurde in Plain wieder gelöscht.")
    print("  3. Die Region stimmt nicht (du nutzt evtl. eine andere Plain-Region).")
    print()
    print("Bitte in Plain prüfen:")
    print("  Settings -> Workspace -> API Keys")
    print("  Ist der Key noch da? Erstelle notfalls einen neuen.")
    sys.exit(1)

if response.status_code == 403:
    print("FEHLER: Plain sagt 'Forbidden' (Status 403).")
    print()
    print("Der Key existiert, aber hat nicht die nötigen Rechte.")
    print()
    print("Bitte in Plain einen neuen Key erstellen MIT diesen Häkchen:")
    print("  - thread:read")
    print("  - customer:read")
    sys.exit(1)

if response.status_code != 200:
    print(f"Unerwarteter Status: {response.status_code}")
    print("Antwort:", response.text[:500])
    sys.exit(1)

data = response.json()

if "errors" in data:
    print("Plain hat einen Fehler zurückgegeben:")
    print(json.dumps(data["errors"], indent=2, ensure_ascii=False))
    print()
    print("Häufig liegt das an fehlenden Permissions. In Plain pruefen:")
    print("  Settings -> Workspace -> API Keys")
    print("  -> Key muss thread:read und customer:read haben.")
    sys.exit(1)

workspace = data.get("data", {}).get("workspace")
if workspace:
    print("ALLES GUT!")
    print(f"  Workspace-Name: {workspace.get('name', '(unbekannt)')}")
    print(f"  Workspace-ID:   {workspace.get('id', '(unbekannt)')}")
    print()
    print("Der Key funktioniert. Jetzt teste ich noch, ob er Threads lesen darf...")

# Zweiter Test: kann er einen einzigen Thread lesen?
print()
threads_query = """
query TestThreads {
  threads(first: 1) {
    edges {
      node { id title }
    }
  }
}
"""

response2 = requests.post(
    PLAIN_API_URL,
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={"query": threads_query},
    timeout=15,
)

if response2.status_code == 200:
    payload = response2.json()
    if "errors" in payload:
        print("Der Key kann sich einloggen, aber NICHT Threads lesen.")
        print("Fehler von Plain:")
        print(json.dumps(payload["errors"], indent=2, ensure_ascii=False))
        print()
        print("Loesung: In Plain einen neuen API-Key erstellen, diesmal")
        print("mit der Berechtigung 'thread:read' (Häkchen setzen).")
        sys.exit(1)
    edges = payload.get("data", {}).get("threads", {}).get("edges", [])
    print(f"Threads-Zugriff: OK ({len(edges)} Beispiel-Thread gefunden)")
    print()
    print("=" * 60)
    print("PERFEKT - dein Key kann alles, was er soll.")
    print("Du kannst jetzt das richtige Export-Skript starten:")
    print("    python3 export_plain_threads.py")
    print("=" * 60)
else:
    print(f"Threads-Test schlug fehl mit Status {response2.status_code}")
    print(response2.text[:500])
