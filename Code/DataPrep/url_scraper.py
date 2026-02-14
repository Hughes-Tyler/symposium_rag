import requests
import csv

url = "https://search-api.swiftype.com/api/v1/public/engines/search.json"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://help.surveymonkey.com"
}

all_results = []

# Loop over pages 1 to 38 
for page in range(1, 39):
    payload = {
        "engine_key": "oHjQNKCNxf9FvxKU2xJG",
        "q": "",
        "filters": {
            "product": ["surveymonkey"]
        },
        "page": page,
        "per_page": 20
    }

    response = requests.post(url, headers=headers, json=payload)
    print(f"Status on page {page}:", response.status_code)

    if response.status_code != 200:
        print(f"Stopping early at page {page} due to status code {response.status_code}")
        break

    data = response.json()
    records = data.get("records", {}).get("page", [])

    if not records:
        print(f"No results found on page {page}, stopping early.")
        break

    all_results.extend(records)

csv_file = "Results/survey_monkey_help_articles.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "URL"])  # Header row
    for record in all_results:
        title = record.get("title", "").strip()
        url_path = record.get("url", "").strip()

        if url_path.startswith("http"):
            full_url = url_path
        else:
            full_url = f"https://help.surveymonkey.com{url_path}"

        writer.writerow([title, full_url])

print(f"\nTotal results fetched: {len(all_results)}")
print(f"Results exported to {csv_file}")
