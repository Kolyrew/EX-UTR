# EX-UTR: Generative architecture that computes 5' 3' UTRs from input protein expression data in target tissues

EX-UTR is a generative model designed to compute 5' and 3' Untranslated Regions (UTRs) from input protein expression data in target tissues. The model aims to facilitate the design of UTR sequences that can modulate gene expression levels, aiding in synthetic biology and therapeutic applications.

## Overview

Untranslated Regions (UTRs) play a crucial role in the post-transcriptional regulation of gene expression. By analyzing protein expression profiles, EX-UTR generates corresponding 5' and 3' UTR sequences that can potentially achieve desired expression levels in specific tissues.

## Repository Structure

- `main.py`: Main script to run the EX-UTR model.
- `Data/`: Contains datasets used for training and evaluation.
- `Parsers/`: Includes scripts for data preprocessing and parsing.
- `checkpoints/ex-utr/`: Directory for saving and loading model checkpoints.


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Kolyrew/EX-UTR.git
   cd EX-UTR
   ```
2. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```
    or 
    ```bash
   pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
   ```
