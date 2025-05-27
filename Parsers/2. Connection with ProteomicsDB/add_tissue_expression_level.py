import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    print('Перед созданием драйвера')
    driver = webdriver.Chrome(options=options)
    print('Драйвер успешно создан')
    wait = WebDriverWait(driver, 15)

    # Читаем ссылки из файла
    with open("GEEEEEEEEEEENEEEEEEEEEEES copy.txt", "r") as f:
        links = [line.strip().split(": ")[-1] for line in f if line.strip() and ": http" in line]

    results = []

    for url in links:
        driver.get(url)

        # Ждем загрузки страницы
        time.sleep(5)

        gene_name = url.split("/")[-2]  # Можно поменять, если нужно другое имя

        # 2. Проверяем наличие "Proteomics" и выбираем радио
        try:
            proteomics_radio = wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR, 'input[type="radio"][value="Proteomics"]'
            )))
            if not proteomics_radio.is_selected():
                proteomics_radio.click()
                time.sleep(2)
        except Exception as e:
            print(f"Proteomics radio not found on {url}: {e}")
            continue

        # 3. Выбираем checkbox "tissue" и убираем все остальные
        try:
            # Сначала убрать все отмеченные чекбоксы, кроме tissue
            all_checkboxes = driver.find_elements(By.CSS_SELECTOR, 'input[type="checkbox"]')
            for cb in all_checkboxes:
                label = driver.find_element(By.CSS_SELECTOR, f'label[for="{cb.get_attribute("id")}"]').text.lower()
                if label == "tissue":
                    if not cb.is_selected():
                        cb.click()
                        time.sleep(1)
                else:
                    if cb.is_selected():
                        cb.click()
                        time.sleep(1)
        except Exception as e:
            print(f"Checkbox handling error on {url}: {e}")
            continue

        # 4. Нажать кнопку "Load Selection"
        try:
            load_button = wait.until(EC.element_to_be_clickable((
                By.XPATH, '//button[contains(.,"Load Selection")]'
            )))
            load_button.click()
            time.sleep(5)
        except Exception as e:
            print(f"Load Selection button not found on {url}: {e}")
            continue

        # 5. Нажать кнопку male (по значению value="male")
        try:
            male_button = wait.until(EC.element_to_be_clickable((
                By.CSS_SELECTOR, 'button[value="male"]'
            )))
            male_button.click()
            time.sleep(5)
        except Exception as e:
            print(f"Male button not found on {url}: {e}")
            # Можно продолжать даже без клика

        # 6. Найти данные тканей и уровни экспрессии
        # В данных на сайте, как правило, есть список тканей и значения
        try:
            # Ткани в html - это обычно <text> или <tspan> с названиями,
            # но т.к. структура может меняться, здесь примерный способ:

            tissues_list = [
                "Brain", "Spinal cord", "Heart", "Thyroid gland", "Lung",
                "Liver", "Pancreas", "Small intestine", "Colon", "Kidney"
            ]

            expression_data = {t: None for t in tissues_list}

            # На странице есть div с классом d3-tip-bodymap,
            # они появляются при наведении — но Selenium не может "навести" и получить opacity>0 легко,
            # поэтому попробуем искать значения в другом месте:
            #
            # Проверим наличие названий тканей и рядом значений.
            # Если нет, то возможно придется парсить JS или svg, но попробуем простой вариант:

            # Найдем все элементы с текстом тканей
            page_source = driver.page_source

            # Ищем в тексте страниц значения вида "TissueName Maximum intensity: X.XX"
            import re
            pattern = r"Organ:\s*([a-zA-Z_ ]+)[^>]*Maximum intensity:\s*([\d\.]+)"
            matches = re.findall(pattern, page_source)

            for organ, value in matches:
                organ = organ.replace("_", " ").strip()
                for tissue in tissues_list:
                    if tissue.lower() == organ.lower():
                        expression_data[tissue] = value

            # Если в matches ничего не нашли - оставим None

        except Exception as e:
            print(f"Error extracting expression data on {url}: {e}")
            expression_data = {t: None for t in tissues_list}

        # 7. Сохраняем результат
        for i, tissue in enumerate(tissues_list):
            if i == 0:
                results.append({
                    "gene_name": gene_name,
                    "tissue": tissue,
                    "expression_level": expression_data.get(tissue)
                })
            else:
                results.append({
                    "gene_name": "",
                    "tissue": tissue,
                    "expression_level": expression_data.get(tissue)
                })

    driver.quit()

    # Записываем в CSV
    df = pd.DataFrame(results)
    df.to_csv("proteomics_expression.csv", index=False)
    print("Данные сохранены в proteomics_expression.csv")

if __name__ == "__main__":
    main()
