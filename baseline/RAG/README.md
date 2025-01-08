# Retrieval-Augmented Generation (RAG)

## Complete Workflow

Here’s an example of the full workflow, demonstrating the steps from data request to final result generation. **Note:** The input path should always match the output path of the previous step.

```bash
# Step 1: Requesting RAG Data
python request.py --task COMA
python request.py --task COST
python request.py --task CSJ

# Step 2: Preprocessing
python baseline/RAG/preprocess.py --dataset_path $REQUEST_FOLDER --output_path $DATA_FOLDER

# Step 3: Splitting
python baseline/RAG/split.py --input_path $DATA_FOLDER --output_path $DATA_FOLDER

# Step 4: Filtering
python baseline/RAG/filter.py --input_path $DATA_FOLDER --output_path $DATA_FOLDER

# Step 5: Embedding
python baseline/RAG/embed.py --path $DATA_FOLDER

# Step 6: Retrieval
python baseline/RAG/retrieve.py --path $DATA_FOLDER --task COMA COST CSJ

# Step 7: Evaluation
python baseline/RAG/evaluate.py --task ALL -prompt BASE --model gpt-3.5-turbo --year 2023 --path $DATA_FOLDER

# Step 8: Get Results
python baseline/RAG/get_results.py --year 2023 --path $DATA_FOLDER

# Optional: Evaluate RAG Quality
python baseline/RAG/relevance_score.py --path $DATA_FOLDER
```

## Script Descriptions

### `request.py`
- **Purpose:** Requests the RAG data for specific tasks.

### `preprocess.py`
- **Purpose:** Preprocesses the RAG datasets by tokenizing text, removing stopwords, and saving the cleaned data into a new file format.
- **File Naming:** The output files will follow the convention `<original_file_name>_preprocessed.jsonl`.

### `split.py`
- **Purpose:** Splits the preprocessed data into smaller chunks, ensuring no chunk exceeds the specified token limit.
- **File Naming:** The output files will follow the convention `<original_file_name>_split.jsonl`.

### `filter.py`
- **Purpose:** Filters the split data based on overlap between the RAG chunks and the question. Only chunks with common words with the question are retained.
- **File Naming:** The output files will follow the convention `<original_file_name>_filter.jsonl`.

### `embed.py`
- **Purpose:** Generates embedding vectors for the filtered data using a specified embedding model.
- **File Naming:** The output files will follow the convention `<original_file_name>_embedding.jsonl`.

### `retrieve.py`
- **Purpose:** Retrieves relevant documents based on a query, using embeddings and reranking techniques.
- **File Naming:** The output file will follow the convention `<task>_rag.jsonl`.

### `evaluate.py`
- **Purpose:** Evaluates the performance of the retrieval results based on the gold answers.

### `get_results.py`
- **Purpose:** Aggregates and formats the evaluation results.

### Optional: `relevance_score.py`
- **Purpose:** Evaluates the relevance of the RAG outputs by calculating the average relevance score and the proportions of scores above 0.8 and below 0.2.
