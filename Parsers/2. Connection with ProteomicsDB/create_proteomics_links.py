import requests
from bs4 import BeautifulSoup
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import json

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--log-level=3')  # меньше логов
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_protein_ids(driver, url):
    protein_ids = set()
    
    driver.get(url)

    try:
        # Ожидаем появления хотя бы одного элемента с data-protein-id (до 10 секунд)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'button[data-protein-id]'))
        )
        buttons = driver.find_elements(By.CSS_SELECTOR, 'button[data-protein-id]')
    except:
        print("⚠️  Элементы не найдены — сохраняем часть HTML:")
        print(driver.page_source[:1000])  # первые 1000 символов для отладки
        return protein_ids

    for button in buttons:
        try:
            protein_id = button.get_attribute('data-protein-id')
            if protein_id:
                protein_ids.add(protein_id)
        except Exception as e:
            print(f"Ошибка при обработке кнопки: {e}")
    
    return protein_ids

# Заголовки для будущих запросов (если будут нужны)
headers = {
    'User-Agent': 'Mozilla/5.0'
}

with open('proteomics_links copy.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_file = 'protein_summary_links.txt'
driver = setup_driver()

try:
    with open(output_file, 'w', encoding='utf-8') as out:
        for line in lines:
            if not line.strip():
                continue
            try:
                gene, url = line.strip().split(': ')
                if 'MDFI' not in gene:
                    continue
                    
                print(f'🔍 Обрабатывается: {gene}')
                
                protein_ids = get_protein_ids(driver, url)
                
                for protein_id in sorted(protein_ids):
                    summary_url = f"https://www.proteomicsdb.org/protein/{protein_id}/summary"
                    out.write(f"{gene} {summary_url}\n")
                
                print(f"✅ Найдено {len(protein_ids)} protein IDs для {gene}")
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Ошибка для {gene}: {e}")
                continue
finally:
    try:
        driver.quit()
    except Exception as e:
        print(f"⚠️ Ошибка при закрытии драйвера: {e}")

print("\n🎉 Готово. Ссылки сохранены в protein_summary_links.txt")
