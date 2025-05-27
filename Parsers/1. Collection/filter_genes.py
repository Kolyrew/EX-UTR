import json

def extract_gene_id_from_url(url):
    # Extract gene_id from URL
    try:
        return url.split('gene_id=')[1].split('&')[0]
    except:
        return None

def filter_genes():
    # Read matching genes
    with open('matching_genes.txt', 'r', encoding='utf-8') as f:
        matching_genes = set(line.strip() for line in f if line.strip() and not line.startswith('='))
    
    # Read and filter utr_5_utrdb_links.json
    with open('utr_5_utrdb_links.json', 'r', encoding='utf-8') as f:
        utr_5_links = json.load(f)
    
    filtered_utr_5_links = [
        link for link in utr_5_links 
        if extract_gene_id_from_url(link['href']) in matching_genes
    ]
    
    # Read and filter utrdb_links.json
    with open('utrdb_links.json', 'r', encoding='utf-8') as f:
        utrdb_links = json.load(f)
    
    filtered_utrdb_links = [
        link for link in utrdb_links 
        if extract_gene_id_from_url(link['href']) in matching_genes
    ]
    
    # Save filtered results
    with open('filtered_utr_5_utrdb_links.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_utr_5_links, f, indent=2, ensure_ascii=False)
    
    with open('filtered_utrdb_links.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_utrdb_links, f, indent=2, ensure_ascii=False)
    
    print(f"Original utr_5_utrdb_links.json entries: {len(utr_5_links)}")
    print(f"Filtered utr_5_utrdb_links.json entries: {len(filtered_utr_5_links)}")
    print(f"Original utrdb_links.json entries: {len(utrdb_links)}")
    print(f"Filtered utrdb_links.json entries: {len(filtered_utrdb_links)}")

if __name__ == "__main__":
    filter_genes() 