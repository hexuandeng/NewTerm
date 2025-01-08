import json
import os
import argparse
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser(description='Edit examples and store results.')
parser.add_argument('--batch_size', type=int, default=1, help='Number to edit at once (default: 1)')
parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
parser.add_argument('--model_type', type=str, choices=['llama', 'qwen'], required=True, help='Type of the model to use (llama or qwen)')
args = parser.parse_args()

output_dir = 'KE/ROME/results'
os.makedirs(output_dir, exist_ok=True)
output_file = f'{output_dir}/{args.model_type}_{args.batch_size}_generate.jsonl'

with open(args.data_path, 'r') as f:
    data = json.load(f)

if args.model_type == 'qwen':
    hparams = ROMEHyperParams.from_hparams('KE/EasyEdit/hparams/ROME/qwen-7b.yaml')
elif args.model_type == 'llama':
    hparams = ROMEHyperParams.from_hparams('KE/EasyEdit/hparams/ROME/llama-7b.yaml')

total_examples = len(data["prompts"])
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

if args.batch_size:
    num_iterations = total_examples // args.batch_size
    with open(output_file, 'w') as f_out:
        for i in range(num_iterations):
            editor = BaseEditor.from_hparams(hparams)
            prompts = data["prompts"][i * args.batch_size:(i + 1) * args.batch_size]
            ground_truth = [None] * args.batch_size
            target_new = data["target_new"][i * args.batch_size:(i + 1) * args.batch_size]
            subject = data["subject"][i * args.batch_size:(i + 1) * args.batch_size]

            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                ground_truth=ground_truth,
                target_new=target_new,
                subject=subject,
                sequential_edit=True
            )

            for prompt, meaning, subj in zip(prompts, target_new, subject):
                inputs = tokenizer(prompt, return_tensors='pt').to(edited_model.device)
                output = edited_model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)

                if output_text.startswith(prompt):
                    output_text = output_text[len(prompt):].strip()

                result = {
                    'prompt': prompt,
                    'output_text': output_text,
                    'subject': subj,
                    'meaning': meaning
                }
                json.dump(result, f_out)
                f_out.write('\n')
                f_out.flush()
else:
    with open(output_file, 'w') as f_out:
        prompts = data["prompts"]
        ground_truth = [None] * total_examples
        target_new = data["target_new"]
        subject = data["subject"]

        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to("cuda")
        for prompt, meaning, subj in zip(prompts, target_new, subject):
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            output = model.generate(**inputs, max_new_tokens=128, num_beams=1, do_sample=False)
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)

            if output_text.startswith(prompt):
                output_text = output_text[len(prompt):].strip()

            result = {
                'prompt': prompt,
                'output_text': output_text,
                'subject': subj,
                'meaning': meaning
            }
            json.dump(result, f_out)
            f_out.write('\n')
            f_out.flush()
