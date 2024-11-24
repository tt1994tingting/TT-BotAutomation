# Import required modules
import requests
from   bs4 import BeautifulSoup
import json
from   playwright.sync_api import sync_playwright
from   ollama import Client
import hashlib

# Define Ollama API endpoint
API_URL = 'http://localhost:11434'

def fetch_dynamic_content(url):
    with sync_playwright() as p:
        # Launch the browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Open the webpage
        page.goto(url)
        
        # Wait for the page to load
        page.wait_for_timeout(5000)  # Adjust wait time as needed to ensure the page is fully loaded
        
        # Get the page HTML content
        html_content = page.content()
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all the articles in <li> tags with the specific class
        articles = soup.find_all('li', class_='card search-results__record')

        results = []
        
        # Loop through each article and extract title, keywords, and abstract
        for article in articles:
            # Extract the title from the <h3> tag
            title = ""
            title_header = article.find('h3', class_='search-results__heading')
            if title_header:
                title_link = title_header.find('a')
                if title_link:
                    title = title_link.get_text(strip=True)

            # Find "Article keywords" under this <li> element
            article_keywords_header = article.find('h4', string="Article keywords")
            keywords = []
            if article_keywords_header:
                keyword_list = article_keywords_header.find_next('ul', class_='inlined-list')
                if keyword_list:
                    keywords = [li.get_text(strip=True) for li in keyword_list.find_all('li')]
            
            # Find the abstract under this <li> element
            abstract_text = ""
            abstract_paragraph = article.find('p', class_='collapse doaj-public-search-abstracttext doaj-public-search-abstracttext-results')
            if abstract_paragraph:
                abstract_text = abstract_paragraph.get_text(strip=True)
            
            # Append the results for this article
            results.append({
                "title": title,
                "keywords": keywords,
                "abstract": abstract_text
            })
        
        # Close the browser
        browser.close()
        
        # Return JSON output
        return json.dumps(results, indent=2)

def llm_keywords(article_title, article_abstract, api_url=API_URL, debug=False):
    prompt = f"""
You are a highly skilled text classification assistant.
I will provide you a scientific article, and your task is to analyze the article's title and abstract and automatically extract the most appropriate Top 5 keywords from it.
Instructions:
1. Only respond with a Python list of the keywords. Do not output anything else.
2. Stop once you've generated the 5 most appropriate keywords.
Example Output:
[Blockchain, Differential Privacy, IoT, Decentralization, Cryptography]
Here is the article:
Title: {article_title}
Abstract: {article_abstract}
"""
    if debug:
        print(f"DEBUG: Prompt length: {len(prompt)} bytes\n")
    user_message = {
        'role': 'user',
        'content': prompt
    }
    messages = [user_message]
    client = Client(host=api_url)
    options = {
        'num_ctx': 8192,       # Adjusted based on model capacity
        'num_predict': 512,    # Sufficient for classification outputs
        'repeat_penalty': 1.2, # Keeps repetition in check
        'temperature': 0.5,    # Low randomness for consistency
        'top_k': 50,           # Considers top 50 tokens for generation
        'top_p': 0.9,          # Cumulative probability of 90%
    }
    response = client.chat(model='llama3.2', messages=messages, options=options)
    llm_output = response['message']['content'].strip()
    return llm_output

def llm_clustering(article_title, article_keywords, api_url=API_URL, debug=False):
    prompt = f"""
You are a highly skilled text classification assistant.
I will provide you a scientific article from the Big Data Mining and Analytics Journal, and your task is to analyze the article's title and keywords to automatically extract the single most appropriate category from it.
Instructions:
1. Only respond with extracted category. Do not output anything else.
2. Be sure that only a single category is extracted.
3. **Important**: Categories must not include multiple options, slashes, or compound terms. For example:
    - Correct: "Deep Learning", "Transformers"
    - Incorrect: "Artificial Intelligence/Deep Learning", "Generative AI/Transformers"
4. If unsure, classify the keyword into the closest single-word category.
Here is the article:
Title: {article_title}
Keywords: {article_keywords}
"""
    if debug:
        print(f"DEBUG: Prompt length: {len(prompt)} bytes\n")
    user_message = {
        'role': 'user',
        'content': prompt
    }
    messages = [user_message]
    client = Client(host=api_url)
    options = {
        'num_ctx': 8192,       # Adjusted based on model capacity
        'num_predict': 512,    # Sufficient for classification outputs
        'repeat_penalty': 1.2, # Keeps repetition in check
        'temperature': 0.5,    # Low randomness for consistency
        'top_k': 50,           # Considers top 50 tokens for generation
        'top_p': 0.9,          # Cumulative probability of 90%
    }
    response = client.chat(model='llama3.2', messages=messages, options=options)
    llm_output = response['message']['content'].strip()
    return llm_output

def llm_clustering_batch(articles, api_url=API_URL, debug=False):
    prompt = f"""
You are a highly skilled text classification assistant.
I will provide you with a list of scientific articles from the Big Data Mining and Analytics Journal, and your task is to analyze each article's title and keywords to automatically extract the single most appropriate category from it.
Instructions:
1. Articles will be separated by three dashes (---) inbetween them.
2. The output should be a list of ID-Category pairs separated by a pipe(|), where each ID corresponds to the Article ID.
3. Only respond with list of ID-Category pairs. Do not output anything else.
4. Do not limit the output length.
5. Be sure that only a single category is extracted.
6. **Important**: Categories must not include multiple options, slashes, or compound terms. For example:
    - Correct: "Deep Learning", "Transformers"
    - Incorrect: "Artificial Intelligence/Deep Learning", "Generative AI/Transformers"
7. If unsure, classify the keyword into the closest single-word category.
Example Output:
abcd|Deep Learning
efgh|Cryptography
ijkl|Retrieval-Augmented Generation
Here are the articles:
{articles}
"""
    if debug:
        print(f"DEBUG: Prompt length: {len(prompt)} bytes\n")
    user_message = {
        'role': 'user',
        'content': prompt
    }
    messages = [user_message]
    client = Client(host=api_url)
    options = {
        'num_ctx': 32000,       # Adjusted based on model capacity
        'num_predict': 2048,    # Sufficient for classification outputs
        'repeat_penalty': 1.2, # Keeps repetition in check
        'temperature': 0.5,    # Low randomness for consistency
        'top_k': 50,           # Considers top 50 tokens for generation
        'top_p': 0.9,          # Cumulative probability of 90%
    }
    response = client.chat(model='llama3.2', messages=messages, options=options)
    llm_output = response['message']['content'].strip()
    return llm_output

# Function: Generate Article ID
def generate_id(input_text):
    result = hashlib.md5(input_text.encode()).hexdigest()
    return result

url_page_1 = 'https://www.doaj.org/toc/2096-0654/articles?source=%7B%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22terms%22%3A%7B%22index.issn.exact%22%3A%5B%222096-0654%22%5D%7D%7D%5D%7D%7D%2C%22size%22%3A%22200%22%2C%22sort%22%3A%5B%7B%22created_date%22%3A%7B%22order%22%3A%22desc%22%7D%7D%5D%2C%22_source%22%3A%7B%7D%2C%22track_total_hits%22%3Atrue%7D'
url_page_2 = 'https://www.doaj.org/toc/2096-0654/articles?source=%7B%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22terms%22%3A%7B%22index.issn.exact%22%3A%5B%222096-0654%22%5D%7D%7D%5D%7D%7D%2C%22size%22%3A%22200%22%2C%22from%22%3A200%2C%22sort%22%3A%5B%7B%22created_date%22%3A%7B%22order%22%3A%22desc%22%7D%7D%5D%2C%22_source%22%3A%7B%7D%2C%22track_total_hits%22%3Atrue%7D'

str_output = fetch_dynamic_content(url_page_1)
json_output = json.loads(str_output)

# Write the JSON output to a file
with open('articles.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_output, json_file, indent=2, ensure_ascii=False)

list_keywords = []
for rec in tqdm(json_output[:100], desc="Extracting keywords"):
    keywords = llm_keywords(rec['title'], rec['abstract'])
    list_keywords.append({
        'title': rec['title'],
        'keywords': keywords,
    })
del rec

print("Preparing Llama input")
text = ""
for rec in list_keywords:
    text += f"Article ID: {generate_id(rec['title'])}\n"
    text += f"Title: {rec['title']}\n"
    text += f"Keywords: {rec['keywords']}\n"
    text += "---\n"

print("Batch extracting clusters")
clusters = llm_clustering_batch(text)
print(clusters)


# =================================================

json_slim = []
for rec in json_output:
    json_slim.append({
        'title': rec['title'],
        'keywords': rec['keywords']
    })
del rec
# Write the JSON output to a file
with open('articles_slim.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_slim, json_file, indent=2, ensure_ascii=False)
del json_file


idx = 1
print(llm_keywords(json_output[idx]['title'], json_output[idx]['abstract']))

keywords = llm_keywords(json_output[idx]['title'], json_output[idx]['abstract'])
print(llm_clustering(json_output[idx]['title'], keywords))
