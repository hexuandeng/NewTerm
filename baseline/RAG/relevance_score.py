import json
import os
import argparse

# Function to calculate and print relevance score statistics
def calculate_relevance_scores(file_path, dataset_name):
    if os.path.exists(file_path):
        all_relevance_scores = []
        # Read file line by line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Extract relevance_scores and term_relevance_scores
                if 'relevance_scores' in data:
                    all_relevance_scores.extend(data['relevance_scores'][:1])

        # Calculate average relevance score and proportions for both
        calculate_statistics(all_relevance_scores, dataset_name, "Relevance")

    else:
        print(f"File not found: {file_path}")

def calculate_statistics(scores, dataset_name, score_type):
    if scores:
        total_scores = len(scores)
        average_score = sum(scores) / total_scores

        greater_than_0_8 = len([score for score in scores if score > 0.8])
        less_than_0_2 = len([score for score in scores if score < 0.2])

        proportion_greater_than_0_8 = greater_than_0_8 / total_scores
        proportion_less_than_0_2 = less_than_0_2 / total_scores

        # Print results
        print(f"\nResults for {dataset_name} ({score_type} Scores):")
        print(f"  - Average {score_type} score: {average_score:.4f}")
        print(f"  - Proportion of {score_type} scores > 0.8: {proportion_greater_than_0_8:.4f}")
        print(f"  - Proportion of {score_type} scores < 0.2: {proportion_less_than_0_2:.4f}")
    else:
        print(f"No {score_type} scores found for {dataset_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate relevance scores for datasets.")
    parser.add_argument('--path', type=str, required=True, help='Directory path containing the dataset files.')

    args = parser.parse_args()

    # Define dataset files
    files = {
        'COMA': os.path.join(args.path, 'COMA_rag.jsonl'),
        'COST': os.path.join(args.path, 'COST_rag.jsonl'),
        'CSJ': os.path.join(args.path, 'CSJ_rag.jsonl')
    }

    # Process each dataset
    for dataset_name, file_path in files.items():
        calculate_relevance_scores(file_path, dataset_name)