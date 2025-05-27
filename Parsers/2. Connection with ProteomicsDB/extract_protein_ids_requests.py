import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

# Заголовки для имитации браузера
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Читаем файл со ссылками
with open('proteomics_links.txt', 'r') as f:
    links = f.readlines()

# Размер batch
BATCH_SIZE = 10

# Обрабатываем ссылки батчами
for i in range(0, len(links), BATCH_SIZE):
    batch = links[i:i+BATCH_SIZE]
    print(f"\nОбработка batch {i//BATCH_SIZE + 1} (гены {i+1}-{min(i+BATCH_SIZE, len(links))})...")

    # Открываем файл для дозаписи результатов
    with open('protein_summary_links.txt', 'a', encoding='utf-8') as outfile:
        with open('gene_to_protein_id.txt', 'a', encoding='utf-8') as idfile:
            # Обрабатываем каждую строку в батче
            for line in batch:
                gene_symbol, search_url = line.strip().split(': ')
                print(f"Processing {gene_symbol}...")

                try:
                    # Получаем страницу поиска
                    response = requests.get(search_url, headers=headers)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Ищем все ссылки на /protein/{id}/summary
                    protein_links = set()
                    pattern = re.compile(r'/protein/(\d+)/summary')

                    for a in soup.find_all('a', href=True):
                        match = pattern.search(a['href'])
                        if match:
                            protein_id = match.group(1)
                            full_link = f"https://www.proteomicsdb.org{a['href']}"
                            protein_links.add(full_link)
                            idfile.write(f"{gene_symbol}: {protein_id}\n")  # сохраняем ID

                    # Записываем ссылки в основной файл
                    if protein_links:
                        outfile.write(f"\n{gene_symbol}:\n")
                        for link in sorted(protein_links):
                            outfile.write(f"{link}\n")
                    else:
                        print(f"No links found for {gene_symbol}")

                    time.sleep(2)  # Пауза между запросами

                except Exception as e:
                    print(f"Error processing {gene_symbol}: {str(e)}")
                    continue

    print(f"Batch {i//BATCH_SIZE + 1} завершен. Результаты сохранены.")
    user_input = input("Нажмите Enter для продолжения или 'q' для выхода: ")
    if user_input.lower() == 'q':
        break

print("\nОбработка завершена! Проверьте файлы:\n - protein_summary_links.txt\n - gene_to_protein_id.txt") 