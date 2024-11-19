
from bs4 import BeautifulSoup
import requests
import re
import sqlite3
from ollama import Client
from datetime import datetime
from configs import *

# Data Processing
def get_arxiv_pdf_links(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # find article ids
            articles_dl = soup.find('dl', id='articles')
            if articles_dl:
                return [
                    f'https://arxiv.org/pdf/{a_tag.get("id")}'
                    for a_tag in articles_dl.find_all('a', title="Abstract")
                    if a_tag.get('id')
                ]
            else:
                print("Didn't find articles'id tags...")
                return []
        else:
            print(f"Request failed with status code: {response.status_code}")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []


def extract_abstract_section(text):
    # start with "Abstract", end with 2 \n
    match = re.search(r'A(?i)bstract\s*(.*?)\n{2,}', text, re.DOTALL)
    
    if match:
        abstract_text = match.group(1).replace('\n', ' ').strip()
        return abstract_text

    return ""


# Bot responses
def bot_response (messages, api_url):
    client = Client(host=api_url)
    response = client.chat(
        model='llama3.2',
        messages=messages,
        options={
            'num_ctx': context_length,
            'num_predict': num_predict,
            'repeat_penalty': repeat_penalty,
        }
    )
    return response['message']['content']


def classify_summaries(api_url, summaries):
    category_counts = {}
    
    prompt_template =  """
    You are a highly specialized AI model that classifies research papers into their most specific yet concise category.
    The category should be as detailed as possible without being overly specific. 
    Only return **one** category name that best fits the content, and avoid using broad categories like "AI" or "Machine Learning".
    Do not include any explanations, additional details, or multiple options. Just provide the single best category.
    
    For example:
    - For research related to human activity recognition, return "Human Activity Recognition".
    - For research on secure software, return "Software Security".
    - For work on image classification or explanability in computer vision, return "Computer Vision".
    
    Now classify the following research summary:
    {content}

    Category:
    """
    
    for url, content in summaries:
        user_message = {
            'role': 'user',
            'content': prompt_template.format(content=content)
        }
        messages = [user_message]
        
        # use model to generate classification
        category = bot_response(messages, api_url).strip()
        
        # keep only tokens as result
        category = category.split('\n')[0].split(" or ")[0].strip()
        
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    return category_counts


def classify_summaries_with_two_layers(api_url, summaries):
    category_counts = {}
    url_classifications = {}  # record classifications for each url

    prompt_template = """
    You are an AI model that classifies research papers into two levels of categories: a broad category (Level 1) and a more specific subcategory (Level 2).
    For each research summary, provide:
    1. A broad category that best fits the content, such as "Machine Learning", "Computer Vision", "Natural Language Processing", "Robotics", etc.
    2. A more specific subcategory within that broad category, such as "Reinforcement Learning" under "Machine Learning".

    Only return the category and subcategory in the format: Level 1: Level 2. Do not provide any additional explanations or options.

    Here are some examples of how to classify research papers:

    Examples:
    - For a paper on improving image classification models, return: Computer Vision: Image Classification
    - For a paper on training agents using rewards, return: Machine Learning: Reinforcement Learning
    - For a paper on summarizing clinical notes, return: Natural Language Processing: Clinical Text Summarization
    - For a paper on new techniques in entity extraction from text, return: Natural Language Processing: Information Extraction
    - For a paper on the use of robots in agriculture, return: Robotics: Agricultural Robots
    - For a paper on adversarial attacks on neural networks, return: Machine Learning: Adversarial Machine Learning
    - For a paper on optimizing hyperparameters in deep learning models, return: Machine Learning: Hyperparameter Tuning
    - For a paper on improving speech recognition, return: Speech Processing: Automatic Speech Recognition
    - For a paper on detecting vulnerabilities in software systems, return: Software Engineering: Software Security

    Now classify the following research summary:
    {content}

    Category:
    """

    for url, content in summaries:
        user_message = {
            'role': 'user',
            'content': prompt_template.format(content=content)
        }
        messages = [user_message]

        # use LLM to get classification, max attempt is 2
        level1, level2 = None, None
        max_attempts = 2
        attempts = 0

        while (level1 is None or level2 is None) and attempts < max_attempts:
            response = bot_response(messages, api_url).strip()
            level1, level2 = parse_two_level_response(response)

            # if failed to classify, add attempt
            if level1 is None or level2 is None:
                attempts += 1
                print(f"Failed to classify summary for URL: {url}, retrying...")

        # failed to classify twice, set as "Others"
        if level1 is None or level2 is None:
            level1, level2 = "Others", "Others"
            print(f"Failed to classify summary for URL: {url} after {max_attempts} attempts. Set to 'Others'.")

        # record response for each url
        url_classifications[url] = (level1, level2)

        # statistic output
        if (level1, level2) in category_counts:
            category_counts[(level1, level2)] += 1
        else:
            category_counts[(level1, level2)] = 1

    # ensure all summaries are classified
    assert len(url_classifications) == len(summaries), "Not all summaries were classified successfully."
    return category_counts, url_classifications

def parse_two_level_response(response):
    """
    covert responses into 2 levels
    """
    if ':' in response:
        level1, level2 = response.split(':', 1)
        return level1.strip(), level2.strip()
    return None, None




# Database
def init_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            url TEXT PRIMARY KEY,
            payload TEXT,
            summary TEXT,
            created_date TEXT
        )
    ''')
    conn.commit()
    return conn

def save_summary_to_db(conn, url, payload, summary):
    cursor = conn.cursor()
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M')
    cursor.execute('''
        INSERT OR REPLACE INTO summaries (url, payload, summary, created_date)
        VALUES (?, ?, ?, ?)
    ''', (url, payload, summary, current_datetime))
    conn.commit()

def get_summary_from_db(conn, url, columns):
    columns_str = ', '.join(columns)
    query = f'SELECT {columns_str} FROM summaries WHERE url = ?'

    cursor = conn.cursor()
    cursor.execute(query, (url,))
    result = cursor.fetchone()
    return result if result else None

def fetch_all_summaries(conn, columns):
    columns_str = ', '.join(columns)
    query = f'SELECT {columns_str} FROM summaries'

    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    return results

def delete_table(conn):
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS summaries')
    conn.commit()
    conn.close()

def init_classification_db():
    conn = sqlite3.connect('classification.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            url TEXT PRIMARY KEY,
            level1 TEXT,
            level2 TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_classification(conn, url, level1, level2):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO classifications (url, level1, level2)
        VALUES (?, ?, ?)
    """, (url, level1, level2))
    conn.commit()
