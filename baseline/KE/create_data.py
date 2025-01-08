import json
import argparse

def process_terms(input_file, output_file):
    output_data = {
        "edit descriptor": "prompt that you want to edit",
        "prompts": [],           # Stores all term prompts
        "ground_truth": None,     # Set to None as required
        "target_new": [],         # Stores all term meanings
        "subject": [],             # Stores term names
        "type": []
    }

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            term_data = json.loads(line)
            
            output_data["prompts"].append(f"What is the meaning of the term '{term_data['term']}'?")
            output_data["subject"].append(term_data['term'])
            output_data["target_new"].append(term_data["meaning"])
            output_data["type"].append(term_data["type"])

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)

    print(f"Data has been processed and saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a terms JSONL file into a structured JSON format.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file')

    args = parser.parse_args()

    process_terms(args.input_file, args.output_file)
