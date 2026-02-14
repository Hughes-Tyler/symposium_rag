# This is the code that grabs the text from all the articles in main.py

import requests
from bs4 import BeautifulSoup
import csv
from tqdm import tqdm
import time

input_csv = "Results/survey_monkey_help_articles.csv"
output_csv = "Results/survey_monkey_help_articles_with_text.csv"

with open(input_csv, newline='', encoding='utf-8') as infile:
    reader = list(csv.DictReader(infile))

with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Title", "URL", "Text"])

    for row in tqdm(reader, desc="Scraping articles", unit="article"):
        url = row["URL"]
        title = row["Title"]

        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; Python Scraper/1.0)"}
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status() 

            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())  

        except Exception as e:
            print(f"Error fetching {url}: {e}")
            text = ""

        writer.writerow([title, url, text])
        time.sleep(0.1) 
