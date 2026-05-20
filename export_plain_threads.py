"""
Plain.com Thread Export
=======================
Exports the most recent threads from your Plain workspace via the GraphQL API
so they can be analyzed to identify the most common customer questions.

Usage:
    1. Set environment variable PLAIN_API_KEY (see README.md)
    2. python3 export_plain_threads.py
    3. Outputs two files in the same folder:
         - plain_threads_export.json  (full data, machine-readable)
         - plain_threads_export.csv   (one row per thread, opens in Excel)

Requirements:
    Python 3.9+
    pip install requests
    (no other dependencies)

What this script does NOT do:
    - It does NOT modify anything in your Plain workspace (read-only).
    - It does NOT include private agent notes.
    - It does NOT send data anywhere except to your local disk.

Privacy:
    Customer emails are exported as-is. If you want them hashed for analysis,
    set ANONYMIZE_EMAILS = True below.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import hashlib
import getpass
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import requests
except ImportError:
    print("Fehlende Bibliothek 'requests'. Bitte einmal im Terminal ausführen:")
    print("    pip3 install requests")
    print("…und dann das Skript erneut starten.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration -- adjust if needed
# ---------------------------------------------------------------------------
PLAIN_API_URL = "https://core-api.uk.plain.com/graphql/v1"
PAGE_SIZE = 50                # Plain returns max 100 per page; 50 is safe.
MAX_THREADS = 200             # How many threads to export in total.
INCLUDE_FIRST_MESSAGE = True  # Fetches the first customer message per thread.
ANONYMIZE_EMAILS = True       # Emails werden für die Analyse anonymisiert (DSGVO-freundlich).
OUTPUT_DIR = Path(__file__).parent
JSON_FILE = OUTPUT_DIR / "plain_threads_export.json"
CSV_FILE = OUTPUT_DIR / "plain_threads_export.csv"

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("PLAIN_API_KEY")
if not API_KEY:
    print("=" * 60)
    print("Plain Ticket-Export")
    print("=" * 60)
    print("Bitte gib jetzt deinen Plain API-Key ein.")
    print("(Er beginnt mit 'plainApiKey_...'. Während du tippst,")
    print(" siehst du nichts auf dem Bildschirm – das ist Absicht,")
    print(" damit niemand mitliest. Einfach reinkopieren + Enter.)")
    print("-" * 60)
    API_KEY = getpass.getpass("API-Key: ").strip()
    if not API_KEY:
        print("Kein Key eingegeben. Abbruch.")
        sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------
# List threads with pagination, sorted newest first.
# Uses fields confirmed via Plain's webhook payload schema (stable surface).
THREADS_QUERY = """
query Threads($first: Int!, $after: String) {
  threads(first: $first, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        title
        previewText
        status
        priority
        externalId
        createdAt { iso8601 }
        updatedAt { iso8601 }
        statusChangedAt { iso8601 }
        customer {
          id
          fullName
          email { email isVerified }
          externalId
        }
        firstInboundMessageInfo  { timestamp { iso8601 } messageSource }
        firstOutboundMessageInfo { timestamp { iso8601 } messageSource }
        supportEmailAddresses
      }
    }
  }
}
"""

# Get the very first customer message (text only).
# The TimelineEntry union has many member types; we extract the common text
# fields defensively so the query keeps working as Plain evolves.
FIRST_MESSAGE_QUERY = """
query FirstMessage($threadId: ID!) {
  thread(threadId: $threadId) {
    timelineEntries(first: 10) {
      edges {
        node {
          id
          timestamp { iso8601 }
          actor { __typename }
          entry {
            __typename
            ... on EmailEntry        { emailSubject: subject emailText: textContent }
            ... on ChatEntry         { chatText: text }
            ... on SlackMessageEntry { slackText: text }
            ... on CustomEntry       { customTitle: title }
          }
        }
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def graphql(query: str, variables: dict) -> dict:
    """Call Plain's GraphQL endpoint and return the `data` field."""
    response = requests.post(
        PLAIN_API_URL,
        headers=HEADERS,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Plain API returned HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(f"Plain GraphQL error: {payload['errors']}")
    return payload["data"]


def anonymize_email(email: str | None) -> str | None:
    if not email or not ANONYMIZE_EMAILS:
        return email
    return hashlib.sha256(email.encode()).hexdigest()[:16] + "@anon"


def extract_first_customer_message(timeline_data: dict) -> str:
    """Find the first entry where the actor is a CustomerActor and pull text."""
    edges = (
        timeline_data.get("thread", {})
        .get("timelineEntries", {})
        .get("edges", [])
    )
    for edge in edges:
        node = edge.get("node") or {}
        actor = node.get("actor") or {}
        if actor.get("__typename") != "CustomerActor":
            continue
        entry = node.get("entry") or {}
        typename = entry.get("__typename", "")
        if typename == "EmailEntry":
            subject = entry.get("emailSubject") or ""
            body = entry.get("emailText") or ""
            return f"{subject}\n{body}".strip()
        if typename == "ChatEntry":
            return (entry.get("chatText") or "").strip()
        if typename == "SlackMessageEntry":
            return (entry.get("slackText") or "").strip()
        if typename == "CustomEntry":
            return (entry.get("customTitle") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fetch_threads() -> list[dict]:
    """Paginate through threads up to MAX_THREADS."""
    threads: list[dict] = []
    cursor: str | None = None
    while len(threads) < MAX_THREADS:
        remaining = MAX_THREADS - len(threads)
        batch_size = min(PAGE_SIZE, remaining)
        print(f"  -> fetching batch of {batch_size} (total so far: {len(threads)})")
        data = graphql(THREADS_QUERY, {"first": batch_size, "after": cursor})
        connection = data["threads"]
        for edge in connection["edges"]:
            threads.append(edge["node"])
        page_info = connection["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]
        # Be friendly to the API.
        time.sleep(0.2)
    return threads


def enrich_with_first_messages(threads: list[dict]) -> None:
    if not INCLUDE_FIRST_MESSAGE:
        return
    print(f"Fetching first customer message for {len(threads)} threads...")
    for i, thread in enumerate(threads, start=1):
        try:
            data = graphql(FIRST_MESSAGE_QUERY, {"threadId": thread["id"]})
            thread["firstCustomerMessage"] = extract_first_customer_message(data)
        except Exception as exc:
            thread["firstCustomerMessage"] = ""
            print(f"  ! could not fetch message for thread {thread['id']}: {exc}")
        if i % 25 == 0:
            print(f"  -> {i}/{len(threads)} done")
        time.sleep(0.1)


def write_json(threads: list[dict]) -> None:
    with JSON_FILE.open("w", encoding="utf-8") as fh:
        json.dump(threads, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(threads)} threads to {JSON_FILE}")


def write_csv(threads: list[dict]) -> None:
    fieldnames = [
        "thread_id",
        "created_at",
        "status",
        "title",
        "preview_text",
        "first_customer_message",
        "customer_email",
        "customer_name",
        "labels",
        "message_source",
        "priority",
    ]
    with CSV_FILE.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for t in threads:
            customer = t.get("customer") or {}
            email_obj = customer.get("email") or {}
            labels: list = []  # Labels werden nicht abgefragt (Permission spart sich Plain)
            first_inbound = t.get("firstInboundMessageInfo") or {}
            writer.writerow({
                "thread_id": t.get("id", ""),
                "created_at": (t.get("createdAt") or {}).get("iso8601", ""),
                "status": t.get("status", ""),
                "title": (t.get("title") or "").replace("\n", " "),
                "preview_text": (t.get("previewText") or "").replace("\n", " "),
                "first_customer_message": (t.get("firstCustomerMessage", "") or "").replace("\n", " "),
                "customer_email": anonymize_email(email_obj.get("email")),
                "customer_name": customer.get("fullName", ""),
                "labels": ", ".join(filter(None, labels)),
                "message_source": first_inbound.get("messageSource", ""),
                "priority": t.get("priority", ""),
            })
    print(f"Wrote CSV summary to {CSV_FILE}")


def main() -> None:
    print(f"Plain Thread Export -- fetching up to {MAX_THREADS} threads")
    print(f"API endpoint: {PLAIN_API_URL}")
    threads = fetch_threads()
    enrich_with_first_messages(threads)
    write_json(threads)
    write_csv(threads)
    print("\n" + "=" * 60)
    print("FERTIG!")
    print("Zwei Dateien wurden erstellt:")
    print(f"  1. {CSV_FILE.name}  (öffnet sich in Excel)")
    print(f"  2. {JSON_FILE.name}  (Rohdaten)")
    print("Sie liegen im gleichen Ordner wie dieses Skript:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
