import json
import numpy as np
import time
import argparse
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
newterm_path = os.path.abspath(os.path.join(current_dir, '../../newterm'))
sys.path.insert(1, newterm_path)

from openai import OpenAI
import cohere

# Initialize OpenAI and Cohere clients
with open(f"config.json", 'r', encoding='utf-8') as f:
    config = json.load(f)
api_keys = config["embedding_key"]
api_base = config["openai_base"]

client = OpenAI(
    api_key=api_keys,
    base_url=api_base,
)
co = cohere.Client(config["cohere_key1"])

# Function to calculate cosine similarity
def cos_sim(A, B):
    A = np.array(A)
    B = np.array(B)
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    return dot_product / (magnitude_A * magnitude_B)

# Function to get embedding vectors
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def retry_rerank(prompt_text, documents, top_n, max_retries=30, delay=10):
    global co
    for attempt in range(max_retries):
        try:
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=prompt_text,
                documents=documents,
                top_n=top_n,
            )
            return rerank_response
        except Exception as e:
            print(f"Rate limit exceeded, retrying in {delay} seconds... ({attempt + 1}/{max_retries})")
            print(f"Error: {e}")
            time.sleep(delay)
        if attempt == 15:
            co = cohere.Client(config["cohere_key2"])

# Function to process the dataset, build prompts, retrieve, and rerank
def process_dataset(folder, task):
    embeddings_file = os.path.join(folder, f"{task}_embedding.jsonl")
    filter_file = os.path.join(folder, f"{task}_filter.jsonl")
    term_file = os.path.join(folder, "TERM_embedding.jsonl")
    term_split_file = os.path.join(folder, "TERM_split.jsonl")
    output_file = os.path.join(folder, f"{task}_rag.jsonl")

    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f_output:
            for line in f_output:
                try:
                    result = json.loads(line)
                    processed_questions.add(result["question"])
                except json.JSONDecodeError:
                    pass

    with open(embeddings_file, "r", encoding="utf-8") as f_embeddings, \
         open(filter_file, "r", encoding="utf-8") as f_filter, \
         open(term_file, "r", encoding="utf-8") as f_terms, \
         open(term_split_file, "r", encoding="utf-8") as f_term_splits, \
         open(output_file, "a", encoding="utf-8") as w_output:

        embeddings_data = {json.loads(line)["question"]: json.loads(line)["embeddings"] for line in f_embeddings}
        filter_data = [json.loads(line) for line in f_filter]
        term_data = {json.loads(line)["term"]: json.loads(line)["embeddings"] for line in f_terms}
        term_split_data = {json.loads(line)["term"]: json.loads(line)["rag"] for line in f_term_splits}

        for line_idx, line in enumerate(filter_data):
            question = line.get("question")
            if question in processed_questions:
                print(f"Skipping already processed question: {question}")
                continue

            print(f"Processing line: {line_idx}")
            docs = line["rag"]
            doc_embeddings = embeddings_data.get(question, [])

            if doc_embeddings:
                question_emb = get_embedding(line["question"])
                asw = [(cos_sim(question_emb, doc_emb), doc) for doc, doc_emb in zip(docs, doc_embeddings)]
                asw.sort(key=lambda x: x[0], reverse=True)

                if asw:
                    rerank_response = retry_rerank(line["question"], [i[1] for i in asw[:500]], top_n=5)
                    relevance_scores = [res.relevance_score for res in rerank_response.results]
                    reranked_sentences = [asw[res.index][1] for res in rerank_response.results]
                else:
                    reranked_sentences, relevance_scores = [], []

            else:
                reranked_sentences, relevance_scores = [], []

            term_docs = term_split_data.get(line.get("term"), [])
            term_embeddings = term_data.get(line.get("term"), [])
            if term_embeddings:
                term_asw = [(cos_sim(question_emb, doc_emb), doc) for doc, doc_emb in zip(term_docs, term_embeddings)]
                term_asw.sort(key=lambda x: x[0], reverse=True)
                term_documents = [i[1] for i in term_asw[:500]]
            else:
                term_documents = []

            if term_documents:
                term_rerank_response = retry_rerank(line["question"], term_documents, top_n=5)
                term_relevance_scores = [res.relevance_score for res in term_rerank_response.results]
                term_reranked_sentences = [term_asw[res.index][1] for res in term_rerank_response.results]
            else:
                term_reranked_sentences, term_relevance_scores = [], []

            result = {
                "rag": reranked_sentences,
                "relevance_scores": relevance_scores,
                "term_rag": term_reranked_sentences,
                "term_relevance_scores": term_relevance_scores,
                "question": line.get("question"),
                "choices": line.get("choices"),
                "gold": line.get("gold"),
                "meaning": line.get("meaning"),
                "split": line.get("split"),
                "term": line.get("term"),
                "type": line.get("type")
            }

            w_output.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed_questions.add(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset for retrieval and reranking.")
    parser.add_argument('--path', type=str, default='data/', help='Path to the folder containing data files.')
    parser.add_argument('--task', type=str, choices=['COMA', 'COST', 'CSJ', 'ALL'], default='ALL', help='Task name (e.g., COMA, COST, CSJ, or ALL).')

    # 解析参数
    args = parser.parse_args()

    # 判断任务
    if args.task == 'ALL':
        for task in ['COMA', 'COST', 'CSJ']:
            process_dataset(args.path, task)
    else:
        process_dataset(args.path, args.task)
