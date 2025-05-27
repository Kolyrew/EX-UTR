import os
import csv
import pandas as pd

# Папка с файлами экспрессии
EXPR_DIR = 'download_temp'
# Файл с UTR-данными
UTR_FILE = 'utrdb_combined_results_with_symbols copy.csv'
# Итоговый файл
OUTPUT_FILE = 'expression_utr_summary.csv'

# Список интересующих тканей
TISSUES = [
    'brain', 'spinal cord', 'heart', 'thyroid gland', 'lung',
    'liver', 'pancreas', 'small intestine', 'colon', 'kidney'
]

# Словарь: gene_name -> {tissue: max_norm_intensity}
gene_expression = {}

# Считываем экспрессию из всех файлов в папке
def collect_expression():
    for fname in os.listdir(EXPR_DIR):
        if not fname.endswith('.csv'):
            continue
        gene_name = fname[:-4]
        path = os.path.join(EXPR_DIR, fname)
        df = pd.read_csv(path, sep=';')
        tissue_map = {}
        for tissue in TISSUES:
            # Ищем первую строку, где Tissue == tissue или Tissue Synonym == tissue
            row = df[(df['Tissue'].str.lower() == tissue) | (df['Tissue Synonym'].str.lower() == tissue)]
            if not row.empty:
                val = row.iloc[0]['Maximum Normalized Intensity']
                tissue_map[tissue] = val
        if tissue_map:
            gene_expression[gene_name] = tissue_map

# Считываем UTR-данные в словарь по gene_symbol
def load_utr_data():
    utr_data = {}
    with open(UTR_FILE, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utr_data[row['gene_symbol']] = row
    return utr_data

def main():
    collect_expression()
    utr_data = load_utr_data()
    
    # Сначала создаем временный список всех строк
    all_rows = []
    for gene, tissue_map in gene_expression.items():
        if gene not in utr_data:
            continue
        utr_row = utr_data[gene]
        first = True
        for tissue in TISSUES:
            if tissue in tissue_map:
                if first:
                    all_rows.append([
                        utr_row['Gene_ID'], utr_row['UTR3_Sequence'], utr_row['UTR5_Sequence'], utr_row['gene_symbol'],
                        tissue.capitalize(), tissue_map[tissue]
                    ])
                    first = False
                else:
                    all_rows.append(['','','','',tissue.capitalize(),tissue_map[tissue]])
    
    # Теперь заполняем пропущенные значения
    current_values = None
    for i, row in enumerate(all_rows):
        if row[0]:  # Если строка содержит Gene_ID
            current_values = row[:4]  # Сохраняем значения Gene_ID, UTR3_Sequence, UTR5_Sequence, gene_symbol
        elif current_values:  # Если строка пустая и у нас есть сохраненные значения
            all_rows[i][:4] = current_values
    
    # Записываем результат в файл
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['Gene_ID','UTR3_Sequence','UTR5_Sequence','gene_symbol','tissue','expression_level'])
        writer.writerows(all_rows)

if __name__ == '__main__':
    main() 