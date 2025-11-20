import requests
from bs4 import BeautifulSoup
import csv

base = "https://medlineplus.gov/druginfo/meds/"
headers = {"User-Agent": "Mozilla/5.0"}

drug_texts = []
keys = []

with open('data/drug_keys.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        keys.append(row[0])

for key in keys:
    url = base + key + ".html"

    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("Failed:", key)
        continue

    soup = BeautifulSoup(r.text, "html.parser")

    # main content block
    container = soup.find("article")
    if not container:
        continue

    text = ""

    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("div", "section-body")] #go by section body not by paragraph
    header = [h.get_text(" ", strip=True) for h in container.find_all("div", "section-title")]
    name = soup.find("h1", "with-also").get_text(" ", strip=True)
    for i in range(len(header)):
        text += header[i] + " "
        text += paragraphs[i] + " "

    drug_texts.append({
        "name": name,
        "text": text
    })

    with open('data/drug_instructions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['name', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(drug_texts)
