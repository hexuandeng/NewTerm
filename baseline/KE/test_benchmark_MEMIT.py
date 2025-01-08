import json
import argparse
from copy import deepcopy
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
newterm_path = os.path.abspath(os.path.join(current_dir, '../../../newterm'))
sys.path.insert(1, newterm_path)
from utils import build_prompt, get_response_open_source
from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Edit examples and evaluate accuracy.')
parser.add_argument('--batch_size', type=int, default=1, help='Number to edit at once (default: 1)')
parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
parser.add_argument('--model_type', type=str, choices=['llama', 'qwen'], required=True, help='Type of the model to use (llama or qwen)')
parser.add_argument('--path', type=str, required=True, help='Path to the COMA, COST, CSJ questions')
parser.add_argument('--year', type=int, default=2023, help='the year of benchmark')
args = parser.parse_args()

batch_sizes = [args.batch_size]
for batch_size in batch_sizes:
    data_path = args.data_path
    with open(data_path, 'r') as f:
        data = json.load(f)

    if args.model_type == 'qwen':
        hparams = MEMITHyperParams.from_hparams('KE/EasyEdit/hparams/MEMIT/qwen-7b.yaml')
    elif args.model_type == 'llama':
        hparams = MEMITHyperParams.from_hparams('KE/EasyEdit/hparams/MEMIT/llama-7b.yaml')

    total_examples = len(data["prompts"])
    num_iterations = total_examples // batch_size


    directory = f'KE/results_{args.year}'
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    for i in range(num_iterations):
        editor = BaseEditor.from_hparams(hparams)
        prompts = data["prompts"][i * batch_size:(i + 1) * batch_size]
        ground_truth = [None] * batch_size
        target_new = data["target_new"][i * batch_size:(i + 1) * batch_size]
        subject = data["subject"][i * batch_size:(i + 1) * batch_size]

        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            sequential_edit=True
        )
        for task in ['COMA','COST','CSJ']:
            file_name = os.path.join(args.path, f'benchmark_{args.year}/{task}_clean.jsonl')
            with open(file_name, 'r', encoding='utf-8') as f, \
                open( f'KE/results_{args.year}/{task}_{args.model_type}-MEMIT{args.batch_size}.jsonl', 'a', encoding='utf-8') as w:
                for line in f:
                    line = json.loads(line.strip())
                    meaning = line['meaning']
                    if meaning in target_new:
                    # Loop through prompt variants
                        for p in range(3):
                            lsb = deepcopy(line)
                            lsb['prompt_id'] = p
                            system, prompt, _ = build_prompt(line, task, p)
                            prompt = [system, prompt]
                            lsb['response'] = get_response_open_source(prompt, args.model_type.capitalize(), edited_model, tokenizer)
                            w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
        