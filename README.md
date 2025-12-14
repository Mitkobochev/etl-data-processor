### Overview
I built a Python pipeline to extract reimbursement decisions from the Danish Medicines Council (*Medicinrådet*). The script scrapes the website, filters for approved drugs, and cleans up the data into a CSV file ready for analysis.

### How it Works
1. **Scraping:** I used the `requests` library to fetch the pages and `BeautifulSoup` to parse the HTML. I noticed the site uses specific database IDs in the URL query parameters, so I targeted those directly to get the correct list of data.
2. **Filtering:** The script looks at the decision status on each card. It keeps only "Anbefalet" (Recommended) or "Delvist anbefalet" (Partially recommended) items, ignoring rejections to keep the dataset focused on positive outcomes.
3. **Extraction:** I used two different approaches depending on how consistent the data was:
    * **Hard Data (Dates, ATC Codes):** These follow predictable formats, so regular expressions work well here.
    * **Adjustable Data (Drug Names):** The "Trade Name" vs. "Generic Name" text in the headers was very inconsistent. Sometimes it uses parentheses, sometimes hyphens, and the order varies.
   
### Why I Used an LLM

I decided to use an LLM (Gemini) specifically for splitting the drug names.
The text in the headers is inconsistent and the word order changes. Hard-coding rules for every variation is messy and hard to maintain if the website changes. The LLM separates them easily without needing complex logic.

**However, I was careful about API calls:**
Instead of calling the LLM for every single row, the script chunk headers and sends them in one request. This keeps the pipeline fast, avoids unnecessary API calls, and scales much better overall.

### Data Quality & Edge Cases
* **Danish Dates:** Month names are in Danish `(like Januar)`, so I added a simple mapping to convert them into standard `YYYY-MM-DD` format.
* **Missing Data:** If a field cant be extracted (either by the scraper or the LLM), its left blank rather than causing the script to fail.
* **Rate Limits:** Small (`sleep`) delays are added between requests to avoid stressing the website.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your Google API key to config.yaml
3. Run the script:
    ```shell
     python main.py
    ```
4. The output will appear default as output.csv