
from bs4 import BeautifulSoup
import requests
import re


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
    match = re.search(r'Abstract\s*(.*?)\n{2,}', text, re.DOTALL)
    
    if match:
        abstract_text = match.group(1).replace('\n', ' ').strip()
        return abstract_text

    return ""

