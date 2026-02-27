### Overview
Python pipeline that extracts reimbursement decisions from the Danish Medicines Council (*Medicinrådet*). Scrapes the site, filters for approved drugs, and outputs a clean CSV ready for analysis.

### How it Works
1. **Scraping:** `requests` fetches the pages, `BeautifulSoup` parses the HTML. The site uses specific database IDs in the URL query params, so I target those directly to pull the right data.
2. **Filtering:** Checks the decision status on each card and keeps only `Anbefalet` (Recommended) or `Delvist anbefalet` (Partially recommended) — rejections are dropped to keep the dataset focused.
3. **Extraction:** Two approaches depending on how consistent the data is:
    - **Structured fields (dates, ATC codes):** Predictable formats, so regex handles these cleanly.
    - **Drug names:** Headers are inconsistent — sometimes parentheses, sometimes hyphens, order varies. Handed off to an LLM (see below).

### Why I Used an LLM
Used Gemini specifically for splitting trade names from generic names. The headers are too inconsistent to hard-code rules for — the format changes and maintaining that logic gets messy fast. The LLM handles the variation cleanly without brittle regex chains.

**To keep it efficient:** instead of calling the API per row, headers are chunked and sent in a single request. Faster, cheaper, and scales better.

### Data Quality & Edge Cases
- **Danish dates:** Month names come in Danish (`Januar`, etc.), so there's a mapping layer to convert them to `YYYY-MM-DD`.
- **Missing fields:** If something can't be extracted — by the scraper or the LLM — it's left blank instead of crashing the pipeline.
- **Rate limiting:** Small `sleep` delays between requests to avoid hammering the site.

### How to Run
1. Install dependencies:
```bash
   pip install -r requirements.txt
```
2. Add your Google API key to `config.yaml`
3. Run:
```bash
   python main.py
```
4. Output saved to `output.csv` by default
