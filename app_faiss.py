import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json
import time
import torch
import faiss

# Load configurations
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

model_path = config['model_path']
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# Load knowledge base
with open('test_base.json', 'r') as file:
    knowledge_base = json.load(file)

knowledge_questions = [qa['question'] for qa in knowledge_base]

# Batch size for embedding
batch_size = 32

def embed_questions_in_batches(questions, batch_size):
    embeddings = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        knowledge_batch = tokenizer(
            batch, max_length=config['max_length'], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            knowledge_outputs = model(**knowledge_batch)
        batch_embeddings = knowledge_outputs.last_hidden_state[:, 0]
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings)

knowledge_embeddings = embed_questions_in_batches(knowledge_questions, batch_size)

# Convert embeddings to numpy array
knowledge_embeddings_np = knowledge_embeddings.cpu().numpy()

# Initialize FAISS index
d = knowledge_embeddings_np.shape[1]
index = faiss.IndexFlatIP(d)                    # Using dot product for similarity
index.add(knowledge_embeddings_np)

def get_answer(query, knowledge_base, index, threshold=0.8):
    st = time.time()
    query_batch = tokenizer([query], max_length=config['max_length'],
                            padding=True, truncation=True, return_tensors='pt')
    t1 = time.time() - st

    st = time.time()
    with torch.no_grad():
        query_output = model(**query_batch)
        query_embedding = query_output.last_hidden_state[:, 0]
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
    t2 = time.time() - st

    st = time.time()
    query_embedding_np = query_embedding.cpu().numpy()
    scores, indices = index.search(query_embedding_np, 1)  # Search for the nearest neighbor
    best_match_index = indices[0][0]
    best_score = scores[0][0]
    t3 = time.time() - st

    if best_score > threshold:
        return knowledge_base[best_match_index]['answer'], t1, t2, t3
    else:
        return "Sorry, I couldn't find a relevant answer to your question.", t1, t2, t3

queries = ["What is the capital of Australia?", "Which element has the atomic number 1?",
           "Who wrote 'The Hobbit'?", "What is Zettabolt known for?", "Who is known as the Father of Modern Zoology?", "Who is the most recent winner of Ballon d'Or?"]

start_time = time.time()
tt = 0
et = 0
st = 0
for query in queries:
    ans, a, b, c = get_answer(query, knowledge_base, index)
    tt += a
    et += b
    st += c
    print(ans)
faiss_time = time.time() - start_time

print("Tokenize time: ", tt/len(queries))
print("Embedding time: ", et/len(queries))
print("Search time: ", st/len(queries))
print(f"FAISS time: {faiss_time:.4f} seconds")
