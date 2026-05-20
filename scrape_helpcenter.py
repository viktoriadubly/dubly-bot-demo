"""
Dubly Help Center Scraper
==========================
Lädt alle Artikel von support.dubly.ai herunter und speichert sie als
helpcenter_articles.json im selben Ordner.

Was es macht:
  1. Lädt die sitemap.xml (offiziell von Dubly bereitgestellt).
  2. Lädt jeden Artikel einzeln und extrahiert Titel + Volltext.
  3. Speichert alles strukturiert als JSON-Datei.

Ausführen (im Terminal):
    pip3 install requests beautifulsoup4
    python3 scrape_helpcenter.py

Dauer: 2-3 Minuten (79 Artikel).
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Fehlende Bibliotheken. Bitte einmal im Terminal:")
    print("    pip3 install requests beautifulsoup4")
    sys.exit(1)

BASE_URL = "https://support.dubly.ai"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
OUTPUT_FILE = Path(__file__).parent / "helpcenter_articles.json"
USER_AGENT = "Mozilla/5.0 (compatible; DublyBotScraper/1.0; Internal Demo)"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def get_article_urls() -> list[str]:
    """Holt alle Artikel-URLs aus der offiziellen Sitemap (via Regex, kein XML-Parser nötig)."""
    print(f"Lade Sitemap: {SITEMAP_URL}")
    resp = session.get(SITEMAP_URL, timeout=30)
    resp.raise_for_status()

    # Sitemap-Format ist simpel: <loc>https://...</loc>. Regex reicht und vermeidet
    # die Abhängigkeit von lxml (BeautifulSoup-XML-Parser braucht das).
    locs = re.findall(r"<loc>\s*([^<]+?)\s*</loc>", resp.text)
    article_urls = [url for url in locs if "/article/" in url]

    print(f"  -> {len(article_urls)} Artikel in Sitemap gefunden")
    return article_urls


def extract_article(url: str) -> dict:
    """Lädt einen Artikel und extrahiert Titel + Volltext + Kategorie."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Titel: bevorzugt <h1>
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""
    if not title:
        title_tag = soup.find("title")
        title = (title_tag.get_text(strip=True) if title_tag else "").split("|")[0].strip()

    # Kategorie: meist als Breadcrumb über dem H1 sichtbar
    # In den Beispiel-Tests ist die erste Zeile der article-Body die Kategorie
    category = ""

    # Hauptinhalt
    main = soup.find("article") or soup.find("main") or soup.body

    if main:
        # Aufräumen: navigation, footer, scripts, related-articles
        for tag in main.find_all(["nav", "footer", "aside", "script", "style"]):
            tag.decompose()

        # Erste Zeile ist oft die Kategorie ("Getting Started", "Billing", ...)
        first_strong = main.find(["span", "div"], class_=lambda c: c and "categor" in c.lower())
        if first_strong:
            category = first_strong.get_text(strip=True)

        # Falls noch keine Kategorie: nimm den ersten kurzen Text vor dem H1
        if not category:
            text_before_h1 = ""
            for child in main.descendants:
                if hasattr(child, "name") and child.name == "h1":
                    break
                if hasattr(child, "string") and child.string:
                    s = child.string.strip()
                    if 3 <= len(s) <= 60 and s != title:
                        text_before_h1 = s
                        break
            category = text_before_h1

        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Aufräumen: Mehrfach-Leerzeilen, Title-Wiederholung am Anfang
    text = re.sub(r"\n{3,}", "\n\n", text)
    # "Related articles" und alles dahinter abschneiden, falls vorhanden
    text = re.split(r"\n+Related articles?\n+", text, flags=re.IGNORECASE)[0].strip()

    return {
        "url": url,
        "title": title,
        "category": category,
        "content": text,
    }


def main() -> None:
    print("=" * 60)
    print("Dubly Help Center Scraper")
    print("=" * 60)

    try:
        urls = get_article_urls()
    except Exception as e:
        print(f"FEHLER beim Sitemap-Laden: {e}")
        sys.exit(1)

    if not urls:
        print("FEHLER: Keine Artikel in der Sitemap gefunden.")
        sys.exit(1)

    print(f"\nLade {len(urls)} Artikel-Volltexte herunter...")
    print("-" * 60)

    result = []
    for i, url in enumerate(urls, start=1):
        try:
            art = extract_article(url)
            result.append(art)
            print(f"  [{i:>3}/{len(urls)}] OK  – {art['title'][:65]}")
        except Exception as e:
            print(f"  [{i:>3}/{len(urls)}] FEHLER – {url}: {e}")
        time.sleep(0.25)  # Rücksicht auf den Server

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_chars = sum(len(a.get("content", "")) for a in result)
    print()
    print("=" * 60)
    print(f"FERTIG: {len(result)} Artikel gespeichert")
    print(f"  Datei: {OUTPUT_FILE.name}")
    print(f"  Insgesamt {total_chars:,} Zeichen Help-Center-Inhalt")
    print()
    print("Nächster Schritt: python3 embed_chunks.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
