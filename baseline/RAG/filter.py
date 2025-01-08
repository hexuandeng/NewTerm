import json
import argparse
import os

def filter_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        print(f"Filtering file: {input_file_path}")
        
        # Read line by line
        for line in infile:
            if not line.strip():  # Skip empty lines
                continue
            try:
                # Parse the JSON object
                data = json.loads(line)
                # Filter the rag chunks
                cleaned_text = data.get('rag', [])
                filtered_chunks = [chunk for chunk in cleaned_text if calculate_overlap(chunk, data.get('question', '')) > 0]
                
                if filtered_chunks:
                    # Update the data object with the filtered chunks
                    data['rag'] = filtered_chunks
                else:
                    data['rag'] = []
                    
                # Write the filtered data to the new file
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {input_file_path}: {e}")

    print(f"File {input_file_path} filtered successfully and written to {output_file_path}")

def calculate_overlap(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(words2)

def main(input_path, output_path):
    # Define the list of preprocessed file names
    file_names = ['CSJ_split.jsonl', 'COST_split.jsonl', 'COMA_split.jsonl']
    
    # Iterate over the list of preprocessed file names
    for file_name in file_names:
        # Construct the full file path
        input_file_path = os.path.join(input_path, file_name)
        output_file_path = os.path.join(output_path, file_name.replace("_split", "_filter"))
        
        # Call the filter function
        filter_file(input_file_path, output_file_path)
    
    print("All files filtered successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter JSONL files based on overlap between 'rag' and 'question'.")
    parser.add_argument('--input_path', type=str, default='data/', help='Path to the directory containing the split files (default: data/).')
    parser.add_argument('--output_path', type=str, default='data/', help='Path to the directory where the filtered files will be saved (default: data/).')

    args = parser.parse_args()
    main(args.input_path, args.output_path)
