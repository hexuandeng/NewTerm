import os
import re
import argparse
import json
from utils import load_open_source, llama_3_response

def generate_sentences_llama(input_file, output_file, num_sentences, model_name):
    model, tokenizer = load_open_source(model_name)

    with open(input_file, 'r', encoding='utf-8') as f:
        terms = [json.loads(line) for line in f]

    with open(output_file, 'w', encoding='utf-8') as w:
        for term_info in terms:
            word = term_info["term"]
            meaning = term_info["meaning"]

            prompt = (
                f'I created a new term "{word}", which means "{meaning}". '
                f'Please generate {num_sentences} different sentences using this term in the format: '
                f'"1. 2." Each sentence should clearly demonstrate the meaning of the term.')

            response = llama_3_response([prompt], model, tokenizer, 64 * num_sentences)
            result = {
                "term": word,
                "meaning": meaning,
                "type": term_info["type"],
                "response": response
            }

            w.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate sentences')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input term.jsonl file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONL file')
    parser.add_argument('--num_sentences', type=int, default=10, help='Number of sentences to generate for each term')
    parser.add_argument('--model', type=str, required=True, help='Path to the LLM')
    args = parser.parse_args()

    tmp = f"tmp_{args.num_sentences}.jsonl"

    generate_sentences_llama(args.input_file, tmp, args.num_sentences, args.model)
    with open(tmp, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            cnt = 0
            data = json.loads(line.strip())
            response_value = data.get('response', '')
            
            match = re.search(r'1\..*', response_value, re.DOTALL)
            term = data['term']
            if match:  
                content = match.group(0)
                text_items = content.split('\n')
                
                for item in text_items:
                    if re.match(r'^\d+\.\s', item.strip()) and cnt < args.num_sentences:
                        new_entry = {
                            'text': item.strip()[3: ].lstrip(),
                            'prefix': f'Please create a sentence using the term "{term}":',
                        }
                        cnt += 1
                        outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')

    os.remove(tmp)
