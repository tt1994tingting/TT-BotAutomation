# Install required modules: pip install -U sentence-transformers faiss-cpu ollama
# NOTE: faiss-gpu cannot be installed through pip on Windows. Use conda instead

import os
import faiss
from   sentence_transformers import SentenceTransformer
from   ollama import Client
import re

# Initialize the SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Ollama client
# NOTE: Ollama is calling Docker ollama image, running on localhost:11434
ollama_client = Client(host="http://localhost:11434")

# Step 1: Load TXT files from the "data" folder
def load_txt_files(folder_path, chunk_size=500, overlap=100, debug=False):
    import re
    docs = []
    file_names = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                raw_text = f.read()
                # Split text into paragraphs for cleaner chunking
                paragraphs = raw_text.split("\n\n")
                for para in paragraphs:
                    # Remove extra spaces and clean up
                    cleaned_text = re.sub(r'\s+', ' ', para).strip()
                    if len(cleaned_text) > 0:
                        # Chunk paragraphs if they are too long
                        chunks = [
                            cleaned_text[i:i + chunk_size]
                            for i in range(0, len(cleaned_text), chunk_size - overlap)
                        ]
                        docs.extend(chunks)
                        file_names.extend([file] * len(chunks))
    if debug:
        print(f"DEBUG: Loaded {len(docs)} chunks from {len(file_names)} files.")
    return docs, file_names

# (Optional) Step 2: Generate embeddings and create FAISS index
def create_faiss_index(documents, index_file="faiss_index.bin", debug=False):
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    dimension = embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    if debug:
        print(f"DEBUG: FAISS index contains {index.ntotal} vectors.")
    
    # Save the index
    faiss.write_index(index, index_file)
    if debug:
        print(f"FAISS index saved to {index_file}.")
    return index, embeddings

# Step 2: Load FAISS index
def load_faiss_index(index_file="faiss_index.bin"):
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        print(f"FAISS index loaded from {index_file}.")
        return index
    else:
        print(f"Index file {index_file} not found. Please create the index first.")
        return None

# Step 3: Retrieve relevant documents
def retrieve_documents(query, index, documents, k=3, debug=False):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    results = [documents[idx] for idx in indices[0]]
    if debug:
        print(f"DEBUG: Retrieved Documents:\n{results}")
    return results

# Step 4: Generate response using Ollama
def generate_response_with_ollama(client, query, context, context_length=2048, num_predict=256, repeat_penalty=1.2, debug=False):
    # Ensure the context is within the allowable length
    context = context[:context_length] if len(context) > context_length else context
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nAnswer:"
    if debug:
        print(f"DEBUG: Generated Prompt:\n{prompt}...")  # Print the first 500 characters of the prompt
    options = {
        'num_ctx': context_length,
        'num_predict': num_predict,
        'repeat_penalty': repeat_penalty,
    }
    response = client.generate(model='llama3.2', prompt=prompt, options=options)
    if not response:
        print("No valid response from Ollama.")
        return "No response generated."
    return response['response'].strip()

# Main RAG system
def rag_system(query, data_folder="data", client=ollama_client, k=3, recreate_index=False, debug=False):
    # Load documents
    documents, file_names = load_txt_files(data_folder, debug=debug)
    
    # Create or load FAISS index
    if recreate_index:
        index, _ = create_faiss_index(documents, debug=debug)
    else:
        index = load_faiss_index()
        if index is None:
            index, _ = create_faiss_index(documents, debug=debug)
    
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, index, documents, k=5, debug=debug)
    
    # Combine retrieved documents as context
    context = "\n---\n".join(retrieved_docs)
    
    # Generate response
    answer = generate_response_with_ollama(client, query, context, debug=debug)
    
    return answer

# Usage
if __name__ == "__main__":
    # Change to True whenever the corpus of documents in ./data changes
    recreaate_index = False
    query = "What is the general trend of research in these files?"
    response = rag_system(query, recreate_index=recreaate_index, debug=True)
    print("Response:", response)
