import os
import re
from   arxiv2text import arxiv_to_text
from   funcs import *
from   tqdm import tqdm

# Specify the data path
data_path = os.path.join(os.getcwd(), "data")

# Check if the folder exists
if not os.path.exists(data_path):
    # Create the folder (including intermediate directories if necessary)
    os.makedirs(data_path)
    print(f"Folder created: {data_path}")
else:
    print(f"Folder already exists: {data_path}")

article_main_url = "https://arxiv.org/list/cs.AI/recent"

test_urls = get_arxiv_pdf_links(article_main_url)

for url in tqdm(test_urls, desc="Extracting PDFs into TXT"):
    #print(f"Processing URL: {url}")
    arxiv_text = arxiv_to_text(url)
    filename = os.path.join(data_path, url.split('/')[-1] + '.txt')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(arxiv_text)

def clean_text(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Step 1: Remove placeholders for images, tables, or figures
    text = re.sub(r'(Figure\s*\d+:|Table\s*\d+:|refer to Figure\s*\d+)', '', text)

    # Step 2: Merge broken lines
    # Replace line breaks that are not followed by another line break
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Step 3: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 4: Preserve section breaks
    # Replace multiple line breaks with two line breaks for readability
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Step 5: Write the cleaned text to a new file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

    return output_path

for file in os.listdir(data_path):
    if file.endswith(".txt"):
        clean_text(os.path.join(data_path, file), os.path.join(data_path, 'clean_'+file))
