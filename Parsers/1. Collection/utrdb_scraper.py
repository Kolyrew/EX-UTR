from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
import json

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    driver = webdriver.Chrome(options=options)
    return driver

def scrape_utrdb():
    driver = setup_driver()
    url = "https://utrdb.cloud.ba.infn.it/cgi-bin/utrdb/uorfquery107.py?UTR_type=3%27UTR&organism=Homo_sapiens.GRCh38.107&search_by=Gene_name_Gene_ID&search_term=&polyA-check=on"
    
    try:
        driver.get(url)
        
        # Wait for the dropdown to be present and select 100 entries
        wait = WebDriverWait(driver, 10)
        dropdown = wait.until(EC.presence_of_element_located((By.NAME, "results_table_length")))
        select = Select(dropdown)
        select.select_by_value("100")
        
        # Wait for the table to update
        time.sleep(1)
        
        all_links = []
        page = 1
        
        while True:
            # Extract links from current page
            links = driver.find_elements(By.CSS_SELECTOR, "td a.btn.btn-success.btn-sm")
            for link in links:
                link_data = {
                    'text': link.text,
                    'href': link.get_attribute('href')
                }
                all_links.append(link_data)
            
            print(f"Processed page {page}, found {len(links)} links")
            
            # Try to click next page
            try:
                next_button = driver.find_element(By.ID, "results_table_next")
                if "disabled" in next_button.get_attribute("class"):
                    break
                next_button.click()
                time.sleep(1)  # Wait for page to load
                page += 1
            except:
                break
        
        # Save results to JSON file
        with open('utrdb_links.json', 'w', encoding='utf-8') as f:
            json.dump(all_links, f, ensure_ascii=False, indent=2)
            
        print(f"Total links collected: {len(all_links)}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_utrdb() 