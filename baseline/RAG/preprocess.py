import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse

def preprocess(dataset_path, output_path, file_names):
    # Tokenize the text and remove stopwords
    # nltk.download('stopwords')
    # nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    # Iterate over the list of file names
    for file_name in file_names:
        input_file_path = f'{dataset_path}{file_name}'
        output_file_path = f'{output_path}{file_name.replace("_clean_search", "_preprocessed")}'
        
        with open(input_file_path, 'r', encoding='utf-8') as file, \
            open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Processing file: {file_name}")
            
            for line in file:
                try:
                    data = json.loads(line)
                    merged_text = ""
                    
                    for item in data.get('rag', []):
                        item.pop('link', None)
                        item.pop('title', None)
                        merged_text += item.get('text', '') + " "
                    
                    merged_text = merged_text.replace('\n', ' ')
                    word_tokens = word_tokenize(merged_text)
                    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
                    cleaned_text = ' '.join(filtered_text)
                    
                    data.pop('rag', None)
                    data['rag'] = cleaned_text.strip()
                    output_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {file_name}: {e}")

        print(f"File {file_name} processed successfully and written to {output_file_path}")

    print("All files processed successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process RAG dataset.')
    parser.add_argument('--dataset_path', type=str, default='data/',
                        help='Specify the dataset path (default is data/)')
    parser.add_argument('--output_path', type=str, default='data/',
                        help='Specify the output path (default is data/)')
    parser.add_argument('--files', type=str, nargs='+', choices=['CSJ', 'COST', 'COMA', 'TERM'],
                        help='Specify which files to process (default is all)')
    
    args = parser.parse_args()

    # Construct file names based on user input
    if args.files:
        file_names = [f"{file}_clean_search.jsonl" for file in args.files]
    else:
        file_names = ['CSJ_clean_search.jsonl', 'COST_clean_search.jsonl', 'COMA_clean_search.jsonl', 'TERM_clean_search.jsonl']

    preprocess(args.dataset_path, args.output_path, file_names)
