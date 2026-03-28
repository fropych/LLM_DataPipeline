---
name: 1-scrape
description: Scrape data from a website using requests + BeautifulSoup with CSS selectors
allowed-tools: Bash, Read, Write
---

Scrape data from a website based on the user's task description.

1. Write `scripts/1_scrape.py`:
   - Send HTTP GET to the target URL
   - Parse HTML with BeautifulSoup
   - Extract text and labels using CSS selectors provided by user
   - Handle pagination / multiple pages if needed
   - Respect rate limits: 1 second delay between requests
   - Map to standard schema: `text`, `label`, `source`, `collected_at`
   - Save to `data/raw/source_scrape.csv`
2. Install deps and run: `uv add requests beautifulsoup4 && uv run python scripts/1_scrape.py`
3. Print: number of rows scraped, source URL
