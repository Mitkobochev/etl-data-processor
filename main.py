import argparse
import logging
import os
import sys

import requests
import yaml
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import json
from typing import List, Dict, Optional
import google.generativeai as genai


class DanishMedicinesETL:
    def __init__(self, api_key: str, chunk_size: int):
        self.logger = logging.getLogger("logfile")
        self.chunk_size = chunk_size
        self.base_url = "https://medicinraadet.dk"
        self.api_endpoint = f"{self.base_url}/anbefalinger-og-vejledninger"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.session = requests.Session()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.month_map = {
            "januar": "01",
            "februar": "02",
            "marts": "03",
            "april": "04",
            "maj": "05",
            "juni": "06",
            "juli": "07",
            "august": "08",
            "september": "09",
            "oktober": "10",
            "november": "11",
            "december": "12",
        }

    def fetch_decisions_list(self, params: Optional[Dict] = None) -> str:
        default_params = {
            "order": "updated desc",
            "currentpageid": "1095",
            "database": "1095",
            "secondary": "1096",
            "category": "",
            "archived": "0",
            "page": "1",
        }
        if params:
            default_params.update(params)

        response = self.session.get(
            self.api_endpoint, headers=self.headers, params=default_params
        )
        response.raise_for_status()
        return response.text

    def get_total_pages(self, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        results_text = soup.find(string=re.compile(r"af\s+\d+\s+resultater", re.I))
        if results_text:
            match = re.search(r"af\s+(\d+)\s+resultater", results_text, re.I)
            if match:
                total_results = int(match.group(1))
                results_per_page = 25
                return (total_results + results_per_page - 1) // results_per_page
        pagination = soup.find(
            "div", class_=lambda x: x and "pagination" in x.lower() if x else False
        )
        max_page = 1
        if pagination:
            page_links = pagination.find_all("a", href=lambda x: x and "page=" in x)
            for link in page_links:
                href = link.get("href", "")
                match = re.search(r"page=(\d+)", href)
                if match:
                    max_page = max(max_page, int(match.group(1)))
        return max_page

    def parse_decision_cards(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        decisions = []
        cards = soup.find_all("div", class_=lambda x: x and "card" in x.lower())
        if not cards:
            cards = soup.find_all("article")
        if not cards:
            links = soup.find_all(
                "a", href=lambda x: x and "/anbefalinger-og-vejledninger/" in x
            )
            cards = [link.find_parent() for link in links if link.find_parent()]

        for card in cards:
            try:
                decision_data = self.extract_decision_from_card(card)
                if decision_data and decision_data.get("status"):
                    decisions.append(decision_data)
            except Exception:
                continue
        return decisions

    def extract_decision_from_card(self, card) -> Optional[Dict]:
        data = {}
        link = card.find(
            "a", href=lambda x: x and "/anbefalinger-og-vejledninger/" in x
        )
        if not link:
            return None

        data["url"] = (
            self.base_url + link["href"]
            if link["href"].startswith("/")
            else link["href"]
        )
        card_text = card.get_text()

        status_patterns = [
            (r"Delvist\s+anbefalet", "Delvist anbefalet"),
            (r"(?<!Ikke\s)(?<!Delvist\s)Anbefalet", "Anbefalet"),
            (r"Ikke\s+anbefalet", "Ikke anbefalet"),
        ]

        data["status"] = None
        for pattern, status_value in status_patterns:
            if re.search(pattern, card_text, re.I):
                data["status"] = status_value
                break

        return data if data.get("status") else None

    def fetch_decision_detail(self, url: str) -> Dict:
        response = self.session.get(url, headers=self.headers)
        response.raise_for_status()
        return self.parse_decision_detail(response.text)

    def parse_decision_detail(self, html: str) -> Dict:
        soup = BeautifulSoup(html, "html.parser")
        data = {}

        main_heading = soup.find("h1")
        heading_text = main_heading.get_text(strip=True) if main_heading else ""

        separators = [" - ", " – ", " — "]
        drug_part = heading_text
        indication_part = None

        for sep in separators:
            if sep in heading_text:
                parts = heading_text.split(sep, 1)
                drug_part = parts[0].strip()
                indication_part = parts[1].strip()
                break
        data["raw_drug_text"] = drug_part
        if indication_part:
            data["indication"] = indication_part
        else:
            usage_label = soup.find(string=re.compile(r"Anvendelse", re.I))
            if usage_label and usage_label.find_parent():
                parent = usage_label.find_parent()
                next_elem = parent.find_next_sibling()
                data["indication"] = (
                    next_elem.get_text(strip=True) if next_elem else None
                )
            else:
                data["indication"] = None

        data["decision_date"] = self.extract_specific_approval_date(soup)
        if not data["decision_date"]:
            data["decision_date"] = self.extract_date(soup)

        data["atc_code"] = self.extract_atc_code(soup)
        return data

    def extract_names_in_chunks(self, text_list: List[str]) -> Dict[str, Dict]:
        if not text_list:
            return {}

        full_results = {}
        total_items = len(text_list)

        self.logger.info(
            f"Starting batch extraction for {total_items} items (Chunk size: {self.chunk_size})..."
        )
        for i in range(0, total_items, self.chunk_size):
            chunk = text_list[i : i + self.chunk_size]
            self.logger.info(
                f"Processing chunk {i // self.chunk_size + 1} ({len(chunk)} items)..."
            )
            json_input = json.dumps(chunk, ensure_ascii=False)

            prompt = f"""
            I will provide a JSON list of Danish medical header texts.
            For each text extract the 'Active Ingredient' (generic name) and 'Trade Name' (brand name).
            If there are multiple drugs, join them with ' + '.

            Return ONLY a JSON object where the keys are the EXACT input strings provided, and the values are objects with keys "active_ingredient" and "trade_name".

            Input List:
            {json_input}
            """
            try:
                response = self.model.generate_content(prompt)
                cleaned_response = (
                    response.text.replace("```json", "").replace("```", "").strip()
                )
                chunk_result = json.loads(cleaned_response)
                full_results.update(chunk_result)
                time.sleep(1)
            except Exception:
                self.logger.exception(f"Failed to process chunk starting at index {i}")
        return full_results

    def extract_specific_approval_date(self, soup) -> Optional[str]:
        rec_div = soup.find("div", id="recommendation")
        if not rec_div:
            return None
        target_text = rec_div.find(string=re.compile(r"Godkendt\s+den", re.I))
        if target_text:
            match = re.search(
                r"Godkendt\s+den\s+(\d{1,2})\.?\s+([a-zA-ZæøåÆØÅ]+)\s+(\d{4})",
                target_text,
                re.I,
            )
            if match:
                day, month_str, year = match.groups()
                month_num = self.month_map.get(month_str.lower(), "01")
                return f"{year}-{month_num}-{day.zfill(2)}"
        return None

    def extract_atc_code(self, soup) -> Optional[str]:
        atc_label = soup.find(string=re.compile(r"ATC-kode", re.I))
        if atc_label and atc_label.find_parent():
            parent = atc_label.find_parent()
            next_elem = parent.find_next_sibling()
            if next_elem:
                return next_elem.get_text(strip=True)

        pattern = re.compile(r"\b[A-Z]\d{2}[A-Z]{2}\d{2}\b")
        match = pattern.search(soup.get_text())
        return match.group(0) if match else None

    def extract_date(self, soup) -> Optional[str]:
        date_patterns = [
            r"\d{1,2}[./\-]\d{1,2}[./\-]\d{4}",
            r"\d{4}[./\-]\d{1,2}[./\-]\d{1,2}",
        ]
        text = soup.get_text()
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def filter_approved_decisions(self, decisions: List[Dict]) -> List[Dict]:
        approved_statuses = ["Anbefalet", "Delvist anbefalet"]
        return [d for d in decisions if d.get("status") in approved_statuses]

    def add_details(self, decisions: List[Dict], delay: float = 1.0) -> List[Dict]:
        enriched_decisions = []
        raw_texts_to_process = set()

        self.logger.info("Phase 1: Fetching HTML details for all approved decisions...")
        for i, decision in enumerate(decisions):
            try:
                if "url" in decision:
                    details = self.fetch_decision_detail(decision["url"])
                    decision.update(details)
                    if decision.get("raw_drug_text"):
                        raw_texts_to_process.add(decision["raw_drug_text"])
                enriched_decisions.append(decision)
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Extracted {i + 1}/{len(decisions)} pages")

                time.sleep(delay)

            except Exception:
                self.logger.exception(f"Error processing {decision.get('url', '')}")
                enriched_decisions.append(decision)
        if raw_texts_to_process:
            self.logger.info("Phase 2: Performing chunk LLM extraction...")
            unique_texts = list(raw_texts_to_process)
            mapping = self.extract_names_in_chunks(unique_texts)
            self.logger.info("Phase 3: Merging LLM results...")
            for decision in enriched_decisions:
                raw_text = decision.get("raw_drug_text")
                if raw_text:
                    if raw_text in mapping:
                        extracted = mapping[raw_text]
                        decision["active_ingredient"] = extracted.get(
                            "active_ingredient", ""
                        )
                        decision["trade_name"] = extracted.get("trade_name", "")
                    else:
                        self.logger.warning(f"LLM missed key: {raw_text}")
                        decision["active_ingredient"] = raw_text
                        decision["trade_name"] = ""
                else:
                    decision["active_ingredient"] = ""
                    decision["trade_name"] = ""

        return enriched_decisions

    def to_dataframe(self, decisions: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(decisions)
        column_mapping = {
            "active_ingredient": "Active Ingredient",
            "trade_name": "Trade Name",
            "atc_code": "ATC Code",
            "decision_date": "Decision Date",
            "indication": "Indication",
        }
        df = df.rename(columns=column_mapping)
        desired_columns = [
            "Active Ingredient",
            "Trade Name",
            "ATC Code",
            "Decision Date",
            "Indication",
        ]
        for col in desired_columns:
            if col not in df.columns:
                df[col] = None
        return df[desired_columns]

    def save_to_csv(self, df: pd.DataFrame, filename: str = "output.csv"):
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        self.logger.info(f"Data saved to {filename}")

    def run_pipeline(self, max_pages: Optional[int] = None) -> pd.DataFrame:
        self.logger.info("Starting ETL pipeline...")
        self.logger.info("Fetching page 1...")
        first_page_html = self.fetch_decisions_list({"page": "1"})
        if max_pages is None:
            max_pages = self.get_total_pages(first_page_html)
            self.logger.info(f"Detected {max_pages} total pages")

        all_decisions = []
        all_decisions.extend(self.parse_decision_cards(first_page_html))

        for page in range(2, max_pages + 1):
            self.logger.info(f"Fetching page {page}/{max_pages}...")
            html = self.fetch_decisions_list({"page": str(page)})
            decisions = self.parse_decision_cards(html)
            if not decisions:
                break
            all_decisions.extend(decisions)
            time.sleep(1)

        self.logger.info(f"Total decisions found: {len(all_decisions)}")
        approved = self.filter_approved_decisions(all_decisions)
        self.logger.info(f"Approved/Partially approved decisions: {len(approved)}")

        if approved:
            approved = self.add_details(approved)

        df = self.to_dataframe(approved)
        return df


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("logfile")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=False,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-cs",
        "--chunk_size",
        type=int,
        required=False,
        default=200,
        help="Chunk size to process LLM query",
    )
    args = parser.parse_args()
    return args


def load_config(config_file: str) -> str:
    if not os.path.exists(config_file):
        raise FileNotFoundError("Yaml configuration file not found!")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    key = config.get("api_key")
    if not key:
        raise ValueError("`api_key` is missing or empty in configuration file")
    return str(key)


def main():
    setup_logging()
    args = parse_arguments()
    api_key = load_config(args.config_file)
    etl = DanishMedicinesETL(api_key=api_key, chunk_size=args.chunk_size)
    df = etl.run_pipeline()
    etl.save_to_csv(df)


if __name__ == "__main__":
    main()
