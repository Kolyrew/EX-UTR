from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
import os
import time
import glob

def setup_driver():
    firefox_options = Options()
    # firefox_options.add_argument('--headless')  # Раскомментируйте для запуска без GUI
    
    # Настройка загрузки файлов
    download_dir = os.path.abspath("D:\\pars\\download_temp")
    os.makedirs(download_dir, exist_ok=True)
    
    firefox_options.set_preference("browser.download.folderList", 2)
    firefox_options.set_preference("browser.download.dir", download_dir)
    firefox_options.set_preference("browser.download.useDownloadDir", True)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv")
    
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=firefox_options)
    return driver

def get_latest_downloaded_file(download_dir):
    """Получает путь к последнему скачанному файлу"""
    list_of_files = glob.glob(os.path.join(download_dir, "*"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def rename_downloaded_file(download_dir, gene_name):
    """Переименовывает последний скачанный файл в формат gene_name.csv"""
    latest_file = get_latest_downloaded_file(download_dir)
    if latest_file:
        new_name = os.path.join(download_dir, f"{gene_name}.csv")
        try:
            if os.path.exists(new_name):
                os.remove(new_name)  # Удаляем существующий файл, если он есть
            os.rename(latest_file, new_name)
            print(f"[INFO] Renamed downloaded file to {gene_name}.csv")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to rename file: {str(e)}")
    return False

def read_urls(filename):
    with open(filename, 'r') as file:
        genes = []
        for line in file:
            if line.strip() and 'https://' in line:
                gene_name = line.strip().split(':')[0].strip()
                url = line.strip().split(' ')[-1]
                genes.append((gene_name, url))
    return genes

def process_url(driver, url, gene_name):
    driver.get(url)
    time.sleep(2)  # Даем время странице загрузиться

    # 1. Выбираем радио-кнопку Proteomics
    try:
        proteomics_radio = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//input[@value='Proteomics']"))
        )
    except TimeoutException:
        print(f"[SKIP] {gene_name}: Proteomics radio not found, skipping.")
        return None

    if not proteomics_radio.is_selected():
        proteomics_radio.click()
        print(f"[INFO] {gene_name}: Selected Proteomics radio button")

    # 2. Выбираем чекбокс tissue
    try:
        tissue_checkbox = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//input[@value='tissue']"))
        )
    except TimeoutException:
        print(f"[SKIP] {gene_name}: Tissue checkbox not found, skipping.")
        return None

    if not tissue_checkbox.is_selected():
        tissue_checkbox.click()
        print(f"[INFO] {gene_name}: Selected tissue checkbox")

    # Снимаем другие чекбоксы, если они выбраны
    other_checkboxes = driver.find_elements(By.XPATH, "//input[@type='checkbox' and @value!='tissue']")
    other_checked = False
    for checkbox in other_checkboxes:
        if checkbox.is_selected():
            other_checked = True
            try:
                checkbox.click()
                print(f"[INFO] {gene_name}: Unchecked other checkbox: {checkbox.get_attribute('value')}")
            except ElementClickInterceptedException:
                driver.execute_script("arguments[0].click();", checkbox)
                print(f"[INFO] {gene_name}: Unchecked other checkbox (via JS): {checkbox.get_attribute('value')}")

    # 3. Нажимаем Load Selection, если были другие галочки
    if other_checked:
        try:
            load_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Load Selection')]"))
            )
            load_button.click()
            print(f"[INFO] {gene_name}: Clicked Load Selection button")
            time.sleep(2)
        except TimeoutException:
            print(f"[SKIP] {gene_name}: Load Selection button not found, skipping.")
            return None

    # 4. Нажимаем кнопку male
    try:
        print(f"[INFO] {gene_name}: Looking for male button...")
        male_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@value='male']"))
        )
        print(f"[INFO] {gene_name}: Found male button, attempting to click...")
        male_button.click()
        print(f"[INFO] {gene_name}: Successfully clicked male button")
        time.sleep(3)  # Ждем загрузки данных
    except TimeoutException:
        print(f"[SKIP] {gene_name}: Male button not found, skipping.")
        return None
    except Exception as e:
        print(f"[ERROR] {gene_name}: Error clicking male button: {str(e)}")
        return None

    # 5. Нажимаем кнопку скачивания и выбираем CSV
    try:
        print(f"[INFO] {gene_name}: Looking for download button...")
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//i[contains(@class, 'mdi-download')]"))
        )
        print(f"[INFO] {gene_name}: Found download button, attempting to click...")
        download_button.click()
        print(f"[INFO] {gene_name}: Clicked download button, waiting for menu...")
        
        # Ждем появления меню с кнопками
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'v-speed-dial__list')]"))
        )
        time.sleep(1)  # Дополнительная пауза для полного появления меню

        # Находим и кликаем на кнопку CSV в выпадающем меню
        csv_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@class='v-btn v-btn--is-elevated v-btn--fab v-btn--has-bg v-btn--round theme--light v-size--small']//span[text()=' CSV ']"))
        )
        print(f"[INFO] {gene_name}: Found CSV button, attempting to click...")
        csv_button.click()
        print(f"[INFO] {gene_name}: Successfully clicked CSV button")
        time.sleep(5)  # Ждем завершения загрузки
        
        # Переименовываем скачанный файл
        download_dir = os.path.abspath("D:\\pars\\download_temp")
        if rename_downloaded_file(download_dir, gene_name):
            return True
        else:
            print(f"[WARN] {gene_name}: Failed to rename downloaded file")
            return None
            
    except TimeoutException:
        print(f"[SKIP] {gene_name}: Download or CSV button not found, skipping.")
        return None
    except Exception as e:
        print(f"[ERROR] {gene_name}: Error during download process: {str(e)}")
        return None

def main():
    driver = setup_driver()
    genes = read_urls('GEEEEEEEEEEENEEEEEEEEEEES copy.txt')

    for gene_name, url in genes:
        process_url(driver, url, gene_name)

    driver.quit()

if __name__ == "__main__":
    main()
