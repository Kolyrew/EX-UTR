from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import re
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('selenium_debug.log'),
        logging.StreamHandler()
    ]
)

def setup_driver():
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Временно отключаем headless режим для отладки
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080')
    return webdriver.Chrome(options=options)

def wait_for_element(driver, by, value, timeout=10):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        logging.error(f"Timeout waiting for element: {value}")
        return None

def extract_protein_ids(driver, gene_symbol, search_url):
    try:
        logging.info(f"Processing {gene_symbol}...")
        driver.get(search_url)
        time.sleep(3)  # Даем странице время на полную загрузку
        
        # Ждем загрузки таблицы и проверяем её наличие
        table = wait_for_element(driver, By.CLASS_NAME, 'dx-datagrid-table')
        if not table:
            logging.error(f"Table not found for {gene_symbol}")
            return set()
        
        # Делаем скриншот для отладки
        driver.save_screenshot(f"debug_{gene_symbol}.png")
        
        # Ищем все строки в таблице
        rows = driver.find_elements(By.CSS_SELECTOR, 'tr.dx-data-row')
        logging.info(f"Found {len(rows)} rows in table for {gene_symbol}")
        
        protein_ids = set()
        
        for row in rows:
            try:
                # Ищем ячейку с геном (пробуем разные селекторы)
                gene_cell = None
                for selector in ['td:nth-child(2)', 'td[data-dx-column-name="Gene Symbol"]', 'td[aria-label*="Gene Symbol"]']:
                    try:
                        gene_cell = row.find_element(By.CSS_SELECTOR, selector)
                        if gene_cell:
                            break
                    except NoSuchElementException:
                        continue
                
                if not gene_cell:
                    logging.warning(f"Could not find gene cell for row in {gene_symbol}")
                    continue
                
                cell_text = gene_cell.text.strip()
                logging.info(f"Found cell text: '{cell_text}' for {gene_symbol}")
                
                if cell_text == gene_symbol:
                    logging.info(f"Found matching row for {gene_symbol}")
                    # Кликаем по строке
                    row.click()
                    time.sleep(2)
                    
                    # Ищем все ссылки на белки
                    links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/protein/"]')
                    logging.info(f"Found {len(links)} protein links for {gene_symbol}")
                    
                    for link in links:
                        href = link.get_attribute('href')
                        match = re.search(r'/protein/(\d+)/summary', href)
                        if match:
                            protein_id = match.group(1)
                            protein_ids.add(protein_id)
                            logging.info(f"Found protein ID: {protein_id} for {gene_symbol}")
            
            except Exception as e:
                logging.error(f"Error processing row for {gene_symbol}: {str(e)}")
                continue
        
        if not protein_ids:
            logging.warning(f"No protein IDs found for {gene_symbol}")
        else:
            logging.info(f"Found {len(protein_ids)} protein IDs for {gene_symbol}")
        
        return protein_ids
    
    except Exception as e:
        logging.error(f"Error processing {gene_symbol}: {str(e)}")
        return set()

def main():
    # Читаем файл со ссылками
    with open('proteomics_links.txt', 'r') as f:
        links = f.readlines()
    
    # Размер batch
    BATCH_SIZE = 5
    
    driver = setup_driver()
    
    try:
        # Обрабатываем ссылки батчами
        for i in range(0, len(links), BATCH_SIZE):
            batch = links[i:i+BATCH_SIZE]
            logging.info(f"\nОбработка batch {i//BATCH_SIZE + 1} (гены {i+1}-{min(i+BATCH_SIZE, len(links))})...")
            
            # Открываем файлы для дозаписи результатов
            with open('protein_summary_links.txt', 'a', encoding='utf-8') as outfile:
                with open('gene_to_protein_id.txt', 'a', encoding='utf-8') as idfile:
                    # Обрабатываем каждую строку в батче
                    for line in batch:
                        gene_symbol, search_url = line.strip().split(': ')
                        protein_ids = extract_protein_ids(driver, gene_symbol, search_url)
                        
                        # Записываем результаты
                        if protein_ids:
                            outfile.write(f"\n{gene_symbol}:\n")
                            for protein_id in sorted(protein_ids):
                                summary_url = f"https://www.proteomicsdb.org/protein/{protein_id}/summary"
                                outfile.write(f"{summary_url}\n")
                                idfile.write(f"{gene_symbol}: {protein_id}\n")
                        else:
                            logging.warning(f"No protein IDs found for {gene_symbol}")
            
            logging.info(f"Batch {i//BATCH_SIZE + 1} завершен. Результаты сохранены.")
            user_input = input("Нажмите Enter для продолжения или 'q' для выхода: ")
            if user_input.lower() == 'q':
                break
    
    finally:
        driver.quit()
    
    logging.info("\nОбработка завершена! Проверьте файлы:\n - protein_summary_links.txt\n - gene_to_protein_id.txt\n - selenium_debug.log")

if __name__ == "__main__":
    main() 