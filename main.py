# Run docker with:
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollamadocker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# After initial container is created, start with:
# docker start ollama

# Import required modules
import json
# from ollama import Clientls
from ollama import Client
from arxiv2text import arxiv_to_text
from funcs import *
from configs import *

# Define the model parameters
context_length = 4096
num_predict = 100
repeat_penalty = 1.2

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

# Define test data set
test_urls = get_arxiv_pdf_links(article_main_url)
for url in test_urls[:10]:
    print(f"for url:{url}")
    arxiv_text = arxiv_to_text(url)
    payload_text = extract_abstract_section(arxiv_text)
    
    if payload_text:
        # print(payload_text)
        # User's message
        user_message = {
            'role': 'user',
            'content': f"Summarize only the key takeaways from the following research paper in no more than 50 words: {payload_text}"
        }

        # Combine messages into a list
        messages = [system_message, user_message]
        res = bot_response (messages, api_url)
        print(res)
    else:
        print("Payload text invalid...")
