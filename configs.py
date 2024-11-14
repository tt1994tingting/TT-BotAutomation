# article url
article_main_url = "https://arxiv.org/list/cs.AI/recent"
# test_url = 'https://arxiv.org/pdf/2411.04991'

# Define API endpoint
api_url = 'http://localhost:11434'

summary_db = 'arxiv_summary.db'


# Define the system message to set the assistant's role and instructions
system_message = {
    'role': 'system',
    'content': 'You are an expert AI research assistant, specialized with extracting useful and actionable insights from research papers.'
}

# Define the model parameters
context_length = 4096
num_predict = 100
repeat_penalty = 1.2