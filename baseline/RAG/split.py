import json
import nltk
from transformers import AutoTokenizer
import argparse
import os

# Ensure nltk is downloaded (for sentence tokenization)
nltk.download('punkt')
tokenizer = AutoTokenizer.from_pretrained('Llama-2-7b-chat-hf', device_map="auto")

def split_paragraph(text, max_length=256, overlap=0, github=False):
    if github:
        sentences = text.split("\n\n")
    else:
        sentences = nltk.sent_tokenize(text)
    
    paragraphs = []
    current_paragraph = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        tokens = tokenizer.tokenize(sentence)
        if len(tokens) > max_length:
            # If the sentence exceeds max length, split it
            for start in range(0, len(tokens), max_length):
                end = start + max_length
                part = tokenizer.convert_tokens_to_string(tokens[start:end])
                if len(tokenizer.tokenize(part)) <= max_length:
                    paragraphs.append(part)
            continue
        
        if current_length + len(tokens) <= max_length:
            current_paragraph.append(sentence)
            current_length += len(tokens)
        else:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = sentences[max(i - overlap, 0): i] + [sentence]
            current_length = sum(len(tokenizer.tokenize(s)) for s in current_paragraph)

    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    return paragraphs

def split(input_path, output_path):
    # Define the list of preprocessed file names 
    preprocessed_file_names = ['CSJ_preprocessed.jsonl', 'COST_preprocessed.jsonl', 'COMA_preprocessed.jsonl', 'TERM_preprocessed.jsonl']
    
    for file_name in preprocessed_file_names:
        input_file_path = os.path.join(input_path, file_name)
        output_file_path = os.path.join(output_path, file_name.replace("_preprocessed", "_split"))
        
        with open(input_file_path, 'r', encoding='utf-8') as file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Splitting file: {file_name}")
            
            for line in file:
                if not line.strip():  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    cleaned_text = data.get('rag', [])
                    
                    # Split the cleaned text into chunks using split_paragraph
                    chunks = split_paragraph(cleaned_text, max_length=256, overlap=0)
                    
                    # Update the data object with the split chunks
                    data['rag'] = chunks
                    
                    # Write the split data to the new file
                    output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                except json.JSONDecodeError as e:
                    print(f"Split error: {e}")

        print(f"File {file_name} split successfully and written to {output_file_path}")

    print("All files split successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split preprocessed JSONL files into smaller chunks.")
    parser.add_argument('--input_path', type=str, default='data/', help='Path to the directory containing the split files (default: data/).')
    parser.add_argument('--output_path', type=str, default='data/', help='Path to the directory where the filtered files will be saved (default: data/).')

    args = parser.parse_args()
    split(args.input_path, args.output_path)
