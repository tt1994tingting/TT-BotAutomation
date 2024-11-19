# Run docker with:
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollamadocker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# After initial container is created, start with:
# docker start ollama

# Import required modules
import json
# from ollama import Clientls
from arxiv2text import arxiv_to_text
from funcs import *
from configs import *

# DB
conn = init_db(summary_db)
# delete_table(conn)


# # Define test data set
# test_urls = get_arxiv_pdf_links(article_main_url)
# for url in test_urls:
#     print(f"Processing URL: {url}")

#     # check if summary exist in DB
#     summary = get_summary_from_db(conn, url,['summary'])
#     if summary:
#         print("Summary found in database!")

#     arxiv_text = arxiv_to_text(url)
#     payload_text = extract_abstract_section(arxiv_text)
    
#     if payload_text:
#         print("########### Abstract ##############")
#         print(payload_text)
#         user_message = {
#             'role': 'user',
#                 'content': f"""Extract the key takeaways from the following research summary in exactly 50 words or less. 
#                             Provide only the summary text without any additional explanation:\n\n{payload_text}"""
#         }
#         messages = [system_message, user_message]
#         summary = bot_response(messages, api_url)
#         print("########### Summary ##############")
#         print(summary)
#         save_summary_to_db(conn, url, payload_text, summary)
#     else:
#         print("Payload text invalid...")


# summary classification
conn = sqlite3.connect(summary_db)
summaries = fetch_all_summaries(conn, ['url','summary'])
print(len(summaries))
# classified_counts = classify_summaries(api_url, summaries[:10])
# print(classified_counts)
category_counts, url_classifications = classify_summaries_with_two_layers(api_url, summaries)

# 打印分类统计结果
print("分类统计结果:")
for (level1, level2), count in category_counts.items():
    print(f"{level1}: {level2} - {count}")

# 打印每个 URL 的分类结果
print("\n每个 URL 的分类结果:")
print(len(url_classifications.items()))
for url, (level1, level2) in url_classifications.items():
    print(f"{url} -> {level1}: {level2}")

# classification DB
classification_conn = init_classification_db()

# 插入分类结果
for url, (level1, level2) in url_classifications.items():
    insert_classification(classification_conn, url, level1, level2)

