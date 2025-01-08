import json
import os
import argparse
from sentence_transformers import SentenceTransformer, util

def compute_similarity(results_file, batch_size, model_name, embedder):
    total_cos_sim = 0
    total_count = 0
    first_200_cos_sim = 0
    all_cos_sim = []
    frequently_words_cos_sim = 0
    new_phrases_cos_sim = 0
    new_words_cos_sim = 0
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            output_text = result['output_text']
            meaning = result['meaning']

            if batch_size > 0:
                if ('qwen' in model_name):
                    output_lines = output_text.split('\u201d')
                    output_lines[0] = output_lines[0]+'\u201d'
                else:
                    output_lines = output_text.split('\n')
                min_meaning_length = min(len(m) for m in meaning) if isinstance(meaning, list) else len(meaning)
                selected_text = output_lines[0]

                if len(selected_text) < min_meaning_length and len(output_lines) > 1:
                    selected_text = selected_text + " " + output_lines[1]
            else:
                selected_text = output_text
            
            generated_embedding = embedder.encode(selected_text, convert_to_tensor=True)
            target_embedding = embedder.encode(meaning, convert_to_tensor=True)
            cosine_similarity = util.cos_sim(generated_embedding, target_embedding).item() if output_text else 0

            total_cos_sim += cosine_similarity
            all_cos_sim.append(cosine_similarity)
            total_count += 1

            if total_count <= 200:
                first_200_cos_sim += cosine_similarity

    if total_count > 100:
        new_words_cos_sim = sum(all_cos_sim[:100])
        new_phrases_cos_sim = sum(all_cos_sim[100:200])
        frequently_words_cos_sim = sum(all_cos_sim[200:300])
        
    first_200_avg_sim = first_200_cos_sim / 200 if total_count >= 200 else (first_200_cos_sim / total_count if total_count > 0 else 0)
    new_words_cos_sim = new_words_cos_sim / 100 if total_count >= 100 else (new_words_cos_sim / total_count if total_count > 0 else 0)
    new_phrases_cos_sim = new_phrases_cos_sim / 100 if total_count >= 100 else (new_phrases_cos_sim / total_count if total_count > 0 else 0)
    frequently_words_cos_sim = frequently_words_cos_sim / 100 if total_count >= 100 else (frequently_words_cos_sim / total_count if total_count > 0 else 0)
    overall_avg_sim = sum(all_cos_sim) / total_count if total_count > 0 else 0

    return overall_avg_sim, first_200_avg_sim, new_words_cos_sim, new_phrases_cos_sim, frequently_words_cos_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute similarity between generated outputs and meanings.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--model_name', type=str, choices=['qwen', 'llama'], required=True, help='Name of the model to use')
    parser.add_argument('--embedder', type=str, required=True, help='Path to the SentenceTransformer model')
    parser.add_argument('--method', type=str, choices=['MEMIT', 'ROME'], default='MEMIT', help='Name of the model to use')
    args = parser.parse_args()

    embedder = SentenceTransformer(args.embedder)
    method = args.method
    if method == 'ROME':
        results_dir = 'KE/ROME/results'
        output_dir = 'KE/ROME/results'
        os.makedirs(output_dir, exist_ok=True)
    if method == 'MEMIT':
        results_dir = 'KE/MEMIT/results'
        output_dir = 'KE/MEMIT/results'
        os.makedirs(output_dir, exist_ok=True)

    if method == 'MEMIT':
        for model in [args.model_name]:
            output_file = f'{output_dir}/{model}_generate_sim_{args.batch_size}.json'
            results_file = f'{results_dir}/{model}_{args.batch_size}_generate.jsonl'

            with open(output_file, 'w') as outfile:
                overall_avg_sim, first_200_avg_sim, new_words, new_phrases, frequently_words = compute_similarity(results_file, args.batch_size, model, embedder)

                outfile.write(json.dumps({
                    "batch_size": args.batch_size,
                    "type_sim": {
                        "new words not deduced": new_words,
                        "new phrases not deduced": new_phrases,
                        "frequently words not deduced": frequently_words,
                        "new words and new phrases": first_200_avg_sim,
                        "overall": overall_avg_sim
                    }
                }) + '\n')

            print(f'Results written to {output_file}')
    else:
        for model in [args.model_name]:
            output_file = f'KE/ROME/results/{model}_generate_sim_{args.batch_size}.json'
            results_file = f'KE/ROME/results/{model}_{args.batch_size}_generate.jsonl'

            with open(output_file, 'w') as outfile:
                overall_avg_sim, first_200_avg_sim, new_words, new_phrases, frequently_words = compute_similarity(results_file, args.batch_size, model, embedder)

                outfile.write(json.dumps({
                    "type_sim": {
                        "new words not deduced": new_words,
                        "new phrases not deduced": new_phrases,
                        "frequently words not deduced": frequently_words,
                        "new words and new phrases": first_200_avg_sim,
                        "overall": overall_avg_sim
                    }
                }) + '\n')

            print(f'Results written to {output_file}')
