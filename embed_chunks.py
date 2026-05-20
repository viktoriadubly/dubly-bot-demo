"""
Embed Chunks für RAG (Google Gemini Version)
============================================
Liest helpcenter_articles.json, teilt jeden Artikel in kleine Stücke (chunks)
und erstellt für jedes Stück einen "Embedding" via Google Gemini.

Speichert das Ergebnis als chunks_with_embeddings.json.

Voraussetzung: Google AI Studio API-Key (aistudio.google.com -> Get API Key).
Kosten: 0 € (Free Tier reicht).

Ausführen:
    pip3 install google-genai
    python3 embed_chunks.py
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Fehlende Bibliothek 'google-genai'. Bitte einmal im Terminal:")
    print("    pip3 install google-genai")
    sys.exit(1)

INPUT_FILE = Path(__file__).parent / "helpcenter_articles.json"
OUTPUT_FILE = Path(__file__).parent / "chunks_with_embeddings.json"
EMBEDDING_MODEL = "gemini-embedding-001"  # Aktuell verfügbares Modell
CHUNK_MAX_CHARS = 1600
CHUNK_OVERLAP_CHARS = 200
BATCH_SIZE = 5  # Klein, weil Free Tier nur 30k Tokens/min erlaubt
BATCH_SLEEP_SECONDS = 4  # Vorsichtig zwischen den Batches

# ---------------------------------------------------------------------------
# API-Key holen
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("=" * 60)
    print("Embed Chunks für RAG (Gemini)")
    print("=" * 60)
    print("Bitte gib deinen Google Gemini API-Key ein.")
    print("(Holst du dir auf aistudio.google.com -> Get API Key.")
    print(" Beginnt mit 'AIza...'. Während du tippst, siehst du nichts.)")
    print("-" * 60)
    API_KEY = getpass.getpass("Gemini API-Key: ").strip()
    if not API_KEY:
        print("Kein Key eingegeben. Abbruch.")
        sys.exit(1)

client = genai.Client(api_key=API_KEY)


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Teilt einen Text in Stücke. Versucht an Absatz-/Satzgrenzen zu trennen."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = pos + max_chars
        if end >= len(text):
            chunks.append(text[pos:].strip())
            break

        cut = -1
        for sep in ["\n\n", "\n", ". ", "! ", "? "]:
            idx = text.rfind(sep, pos + max_chars // 2, end)
            if idx > cut:
                cut = idx + len(sep)
        if cut <= 0:
            cut = end

        chunks.append(text[pos:cut].strip())
        pos = max(cut - overlap, pos + 1)

    return [c for c in chunks if c]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Schickt eine Batch von Texten an Gemini und gibt die Embeddings zurück."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [emb.values for emb in result.embeddings]


def main() -> None:
    if not INPUT_FILE.exists():
        print(f"FEHLER: {INPUT_FILE.name} nicht gefunden.")
        print("Bitte erst scrape_helpcenter.py ausführen.")
        sys.exit(1)

    with INPUT_FILE.open(encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Geladen: {len(articles)} Artikel aus {INPUT_FILE.name}")

    # In Chunks aufteilen
    chunks: list[dict] = []
    for art in articles:
        content = art.get("content", "")
        if not content:
            continue
        prefix = ""
        if art.get("category"):
            prefix += f"Category: {art['category']}\n"
        if art.get("title"):
            prefix += f"Article: {art['title']}\n\n"

        for i, piece in enumerate(chunk_text(content)):
            chunks.append({
                "id": f"{art.get('url', 'unknown')}#{i}",
                "url": art.get("url", ""),
                "title": art.get("title", ""),
                "category": art.get("category", ""),
                "chunk_index": i,
                "text": prefix + piece,
            })

    print(f"Aufgeteilt in {len(chunks)} Chunks (~{sum(len(c['text']) for c in chunks):,} Zeichen)")
    print()
    print(f"Erstelle Embeddings via Gemini (Modell: {EMBEDDING_MODEL})")
    print(f"Batch-Größe: {BATCH_SIZE}")
    print("-" * 60)

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start:batch_start + BATCH_SIZE]
        try:
            embeddings = embed_batch([c["text"] for c in batch])
            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb
            done = min(batch_start + BATCH_SIZE, len(chunks))
            print(f"  [{done:>3}/{len(chunks)}] Batch OK")
        except Exception as e:
            err_str = str(e)
            wait = 35 if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) else 10
            print(f"  FEHLER bei Batch ab {batch_start}: {err_str[:140]}")
            print(f"  Versuche es in {wait} Sekunden nochmal...")
            time.sleep(wait)
            try:
                embeddings = embed_batch([c["text"] for c in batch])
                for chunk, emb in zip(batch, embeddings):
                    chunk["embedding"] = emb
                print(f"  [{batch_start + len(batch):>3}/{len(chunks)}] Batch OK (Retry)")
            except Exception as e2:
                print(f"  Endgültiger Fehler: {e2}")
                print("  Tipp: BATCH_SIZE im Skript-Header auf 1 setzen, dann läuft es garantiert.")
                sys.exit(1)
        time.sleep(BATCH_SLEEP_SECONDS)

    chunks_with_emb = [c for c in chunks if "embedding" in c]

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks_with_emb, f, ensure_ascii=False)

    file_size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024

    print()
    print("=" * 60)
    print(f"FERTIG: {len(chunks_with_emb)} Chunks mit Embeddings gespeichert")
    print(f"  Datei: {OUTPUT_FILE.name} ({file_size_mb:.1f} MB)")
    print()
    print("Nächster Schritt: python3 dubly_bot_rag.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
