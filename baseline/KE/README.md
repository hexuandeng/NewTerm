# Knowledge Editing (KE)

## Step 1: Create Data

This script processes the `new_terms.jsonl` file into a structured JSON format, making it suitable for editing purposes.

### Usage

To run the script, execute the following command:

```bash
python baseline/KE/create_data.py --input_file benchmark_2023/new_terms.jsonl --output_file baseline/KE/train.json
```

## Step 2: Clone EasyEdit Repository

To set up the project, clone the EasyEdit repository into the `baseline/KE` directory by using the following command:

```bash
git clone https://github.com/zjunlp/EasyEdit.git baseline/KE/EasyEdit/
```

After cloning the EasyEdit repository, you need to update the model names in the configuration files. Open the relevant YAML files and modify the parameters to suit your model settings.

Finally, ensure the directory structure matches the following layout. 

```bash
baseline
├── KE
│   ├── create_data.py
│   ├── compute_sim.py
│   └── EasyEdit
│       ├── evaluate_memory_performance.py
│       ├── test_generate_ROME.py
│       ├── test_benchmark_ROME.py
│       ├── test_generate_MEMIT.py
│       └── test_benchmark_MEMIT.py
```


## Step 3: Evaluate Model Accuracy (Benchmark)

This script evaluates the model's accuracy on a **benchmark** while editing a specified number of words per batch.

### Usage

To run the script, use the following command:

```bash
python baseline/KE/EasyEdit/test_benchmark_MEMIT.py --batch_size <your_batch_size> --data_path <your/path/to/train.json> --model_path <your/path/to/model> --model_type <llama|qwen> --path <your/path/to/benchmark>
```

### Parameters

- `--batch_size`: The number of words to edit at once (default: 1).
- `--data_path`: Path to the input data file generated in Step 1 (e.g., `train.json`).
- `--model_path`: Path to the directory containing the model files.
- `--model_type`: Type of the model to use, either `llama` or `qwen`.
- `--path`: Path to the directory containing COMA, COST, and CSJ.

### Example

```bash
python baseline/KE/EasyEdit/test_benchmark_MEMIT.py --batch_size 50 --data_path baseline/KE/train.json --model_path LLMs/Qwen-7B-Chat --model_type qwen --path benchmark_2023/
```

This command will evaluate the model's performance on the benchmark after editing 50 words per batch.

### Output

The results will be saved in the `KE/results` directory, with a filename indicating the model type and batch size, such as `qwen_50_benchmark.json`. The output file will contain the average accuracy for different tasks (COMA, COST, CSJ).


## Step 4: Evaluate Model Accuracy (Cosine Similarity)

This module consists of two scripts: one for editing prompts and generating outputs, and another for computing the similarity between those outputs and their associated meanings.

### Part 1: Generate Outputs

This script edits prompts from a specified data file using a chosen model and stores the generated outputs in a structured JSON Lines format.

#### Usage

To run the script, use the following command:

```bash
python baseline/KE/EasyEdit/test_generate_MEMIT.py --batch_size <your_batch_size> --data_path <your/path/to/train_data.json> --model_path <your/path/to/model> --model_type <llama|qwen>
```

#### Arguments

- `--batch_size`: Number of examples to edit at once (default: 1; use 0 for no edits).
- `--data_path`: Path to the input data file (JSON format).
- `--model_path`: Path to the model directory.
- `--model_type`: Specify the model type to use, either `llama` or `qwen`.

#### Output

The script generates a file named `<model_type>_generate_<batch_size>.jsonl` in the `baseline/KE/results` directory, containing the following fields for each edited prompt:

- `prompt`: The original input prompt.
- `output_text`: The generated output from the model.
- `subject`: The subject of the prompt.
- `meaning`: The target meaning associated with the prompt.

#### Example Command

```bash
python baseline/KE/EasyEdit/test_generate_MEMIT.py --batch_size 50 --data_path baseline/KE/train.json --model_path /path/to/your/qwen --model_type qwen
```

This command will process the input data in batches of 50 and store the results in `baseline/KE/results/qwen_generate_50.jsonl`.


### Part 2: Evaluate Similarity

This script computes the cosine similarity between the generated outputs and their corresponding meanings, using a specified SentenceTransformer model.

#### Usage

To run the script, use the following command:

```bash
python baseline/KE/compute_sim.py --batch_size <your_batch_size> --model_name <qwen|llama> --embedder <path/to/all-mpnet-base-v2>
```

#### Arguments

- `--batch_size`: Batch size used in processing (required).
- `--model_name`: Name of the model, either `qwen` or `llama` (required).
- `--embedder`: Path to the SentenceTransformer model (required).

#### Output

The script generates a file named `<model_name>_generate_sim_<batch_size>.json` in the `baseline/KE/results` directory, containing the following similarity metrics:

- `new words not deduced`: Average similarity for new words.
- `new phrases not deduced`: Average similarity for new phrases.
- `frequently words not deduced`: Average similarity for frequently used words.
- `new words and new phrases`: Average similarity for the first 200 records.
- `overall`: Overall average similarity across all records.

#### Example Command

```bash
python baseline/KE/compute_sim.py --batch_size 50 --model_name qwen --embedder /path/to/your/all-mpnet-base-v2
```

This command will compute the similarity metrics for the outputs generated with a batch size of 50, using the Qwen model and the specified SentenceTransformer. The results will be saved in `baseline/KE/results/qwen_generate_sim_50.json`.


## Step 5: Evaluate Model Memory and Performance with Increasing Edit Batch Size

### Overview

This module evaluates the model's responses as the number of edits increases, storing results in a structured JSON Lines format. This allows for a detailed analysis of model performance based on the number of edits applied.

To evaluate the model's memory and performance, use the following command:

```bash
python baseline/KE/EasyEdit/evaluate_memory_performance.py --model_name <qwen|llama> --model_path <your/path/to/model> --data_path <your/path/to/train.json>
```

#### Arguments

- `--model_name`: Name of the model to use (`qwen` or `llama`).
- `--model_path`: Path to the model directory, which includes the tokenizer and hyperparameter files.
- `--data_path`: Path to the input data file in JSON format containing prompts and associated data.

#### Output

The script generates two files in the `baseline/KE/results` directory:

- `<model_name>_latest25.jsonl`: Contains the results of the latest 25 edits, including fields for prompts, output texts, subjects, and meanings.
- `<model_name>_first25.jsonl`: Contains results for the first 25 prompts, allowing for comparison against later edits.

#### Example Command

```bash
python baseline/KE/EasyEdit/evaluate_memory_performance.py --model_name qwen --model_path /path/to/your/qwen --data_path baseline/KE/train.json
```
