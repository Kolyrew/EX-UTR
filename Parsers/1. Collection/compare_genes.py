import json

def extract_gene_id_from_url(url):
    # Extract gene_id from URL
    try:
        return url.split('gene_id=')[1].split('&')[0]
    except:
        return None

def compare_genes():
    # Read both JSON files
    with open('utr_5_utrdb_links.json', 'r', encoding='utf-8') as f:
        utr_5_links = json.load(f)
    
    with open('utrdb_links.json', 'r', encoding='utf-8') as f:
        utrdb_links = json.load(f)
    
    # Extract gene IDs from both files
    utr_5_genes = set()
    for link in utr_5_links:
        gene_id = extract_gene_id_from_url(link['href'])
        if gene_id:
            utr_5_genes.add(gene_id)
    
    utrdb_genes = set()
    for link in utrdb_links:
        gene_id = extract_gene_id_from_url(link['href'])
        if gene_id:
            utrdb_genes.add(gene_id)
    
    # Find matching genes
    matching_genes = utr_5_genes.intersection(utrdb_genes)
    
    # Save matching genes to text file
    with open('matching_genes.txt', 'w', encoding='utf-8') as f:
        f.write("Matching genes between utr_5_utrdb_links.json and utrdb_links.json:\n")
        f.write("=" * 80 + "\n\n")
        for gene in sorted(matching_genes):
            f.write(f"{gene}\n")
    
    print(f"Found {len(matching_genes)} matching genes")
    print(f"Results saved to matching_genes.txt")

if __name__ == "__main__":
    compare_genes() 