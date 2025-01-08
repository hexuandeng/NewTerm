import json
import time
import random
import os
import argparse
from openai import OpenAI
import re

def embed(output_path, max_retries=30, retry_delay=5):
    with open(f"config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    api_keys = config["embedding_key"]
    api_base = config["openai_base"]

    client = OpenAI(
        api_key=api_keys,
        base_url=api_base,
    )

    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(input=[text], model=model)
                return response.data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay + random.uniform(0, 5)  # Random delay to avoid throttling
                    print(f"Retrying in {wait_time:.2f} seconds due to: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise

    file_names = ['TERM_split.jsonl', 'CSJ_filter.jsonl', 'COST_filter.jsonl', 'COMA_filter.jsonl']
    for file_name in file_names:
        new_file_name = re.sub(r'(_filter|_split)', '_embedding', file_name)
        input_file_path = os.path.join(output_path, file_name)
        output_file_path = os.path.join(output_path, new_file_name)

        # Step 1: Load existing embeddings to check for duplicates
        existing_questions = set()
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r', encoding='utf-8') as output_file:
                for line in output_file:
                    try:
                        data = json.loads(line)
                        if 'question' in data:
                            existing_questions.add(data['question'])
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in existing file {file_name}: {e}")

        total_lines = 0
        successful_lines = 0
        skipped_lines = 0
        failed_lines = 0

        mode = 'w' if 'TERM' in file_name else 'a'
        # Step 2: Process the filter file
        with open(input_file_path, 'r', encoding='utf-8') as file, \
             open(output_file_path, mode, encoding='utf-8') as output_file:  # 'a' mode to append
            print(f"Embedding file: {file_name}")

            for line in file:
                total_lines += 1
                try:
                    data = json.loads(line)
                    question = data.get('question')

                    # Skip if the question already has embeddings
                    if question in existing_questions:
                        skipped_lines += 1
                        continue

                    # Otherwise, process the embedding
                    chunks = data.get('rag', [])
                    embeddings = [get_embedding(chunk) for chunk in chunks]
                    data.pop('rag', None)
                    data['embeddings'] = embeddings
                    output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                    successful_lines += 1
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_name}: {e}")
                    failed_lines += 1
                except Exception as e:
                    print(f"Failed to process line in {file_name}: {e}")
                    failed_lines += 1

        print(f"Finished embedding {file_name}: {successful_lines}/{total_lines} lines succeeded, {failed_lines} failed, {skipped_lines} skipped.")

    print("All files embedded successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for questions in JSONL files.")
    parser.add_argument('--path', type=str, default='data/', help='Path to the directory where the files are located (default: data/).')

    args = parser.parse_args()
    embed(args.path)
