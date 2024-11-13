# Run docker with:
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollamadocker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# After initial container is created, start with:
# docker start ollama

# Import required modules
import requests
import json
from   ollama import Client
from   bs4 import BeautifulSoup
from   arxiv2text import arxiv_to_text

# Define API endpoint
api_url = 'http://localhost:11434'

# Define the model parameters
context_length = 4096
num_predict = 100
repeat_penalty = 1.2

# Function: Parse HTML
def parse_html(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text

# Define test data set
test_url = 'https://arxiv.org/pdf/2411.04991'
arxiv_text = arxiv_to_text(test_url)
payload_text = arxiv_text[arxiv_text.find('Abstract\n'):arxiv_text.find('Keywords:')].replace('Abstract\n', '').replace('\n', ' ').strip()

# Define the system message to set the assistant's role and instructions
system_message = {
    'role': 'system',
    'content': 'You are an expert AI research assistant, specialized with extracting useful and actionable insights from research papers.'
}

# User's message
user_message = {
    'role': 'user',
    'content': f"Summarize only the key takeaways from the following research paper in no more than 50 words: {payload_text}"
}

# Combine messages into a list
messages = [system_message, user_message]

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

print(response['message']['content'])
