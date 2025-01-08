import json
import os
import random
import argparse
from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
from transformers import AutoTokenizer

def evaluate_model_memory_performance(model_name, model_path, data_path, output_dir):
    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    prompts = data["prompts"][:200]
    target_new = data["target_new"][:200]
    subject = data["subject"][:200]
    ground_truth = [None] * 200
    
    combined = list(zip(prompts, target_new, ground_truth, subject))
    random.shuffle(combined)
    prompts, target_new, ground_truth, subject = zip(*combined)
    
    prompts = list(prompts)
    target_new = list(target_new)
    ground_truth = list(ground_truth)
    subject = list(subject)
    # Initialize model parameters based on the selected model
    if model_name == 'llama':
        hparams = MEMITHyperParams.from_hparams('KE/EasyEdit/hparams/MEMIT/llama-7b.yaml')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name == 'qwen':
        hparams = MEMITHyperParams.from_hparams('KE/EasyEdit/hparams/MEMIT/qwen-7b.yaml')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    os.makedirs(output_dir, exist_ok=True)

    output_file = f'{output_dir}/{model_name}_latest25.jsonl'
    output_file_1 = f'{output_dir}/{model_name}_first25.jsonl'

    with open(output_file, 'w') as f_out, open(output_file_1, 'w') as f_rem: 
        for i in range(8):
            part_prompts = prompts[:(i+1)*25]
            part_ground_truth = ground_truth[:(i+1)*25]
            part_target_new = target_new[:(i+1)*25]
            part_subject = subject[:(i+1)*25]
            editor = BaseEditor.from_hparams(hparams)

            metrics, edited_model, _ = editor.edit(
                prompts=part_prompts,
                ground_truth=part_ground_truth,
                target_new=part_target_new,
                subject=part_subject,
                sequential_edit=True
            )

            for prompt, meaning, subj in zip(prompts[i*25:(i+1)*25], target_new[i*25:(i+1)*25], subject[i*25:(i+1)*25]):
                inputs = tokenizer(prompt, return_tensors='pt').to(edited_model.device)
                output = edited_model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)

                if output_text.startswith(prompt):
                    output_text = output_text[len(prompt):].strip()

                result = {
                    'batch_size': (i + 1) * 25,
                    'prompt': prompt,
                    'output_text': output_text,
                    'subject': subj,
                    'meaning': meaning
                }
                json.dump(result, f_out)
                f_out.write('\n')
                f_out.flush()  

            for prompt, meaning, subj in zip(prompts[:25], target_new[:25], subject[:25]):
                inputs = tokenizer(prompt, return_tensors='pt').to(edited_model.device)
                output = edited_model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)

                if output_text.startswith(prompt):
                    output_text = output_text[len(prompt):].strip()

                result = {
                    'batch_size': (i + 1) * 25,
                    'prompt': prompt,
                    'output_text': output_text,
                    'subject': subj,
                    'meaning': meaning
                }
                json.dump(result, f_rem)
                f_rem.write('\n')
                f_rem.flush()  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model memory and performance based on edits.')
    parser.add_argument('--model_name', type=str, choices=['qwen', 'llama'], required=True, help='Model name to use (qwen or llama).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file (JSON format).')
    args = parser.parse_args()
    
    output_dir = 'KE/MEMIT/results'
    evaluate_model_memory_performance(args.model_name, args.model_path, args.data_path, output_dir)
