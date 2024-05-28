import os
import json
import argparse
from copy import deepcopy
from utils import MultiChat, build_prompt, get_few_shot, get_response_open_source, load_open_source

def test_open_source(task, prompt_type, model_name, year):
    model, tokenizer = load_open_source(model_name)
    if prompt_type == '_few_rand':
        few = []
        with open(f'benchmark_{year}/{task}_clean.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                few.append(line)
    
    with open(f'benchmark_{year}/{task}_clean.jsonl', 'r', encoding='utf-8') as f,\
        open(f'newterm/results_{year}/{task}_{model_name.split("/")[-1]}{prompt_type}.jsonl', 'w', encoding='utf-8') as w:
        for line in f:
            line = json.loads(line.strip())
            for p in range(3):
                lsb = deepcopy(line)
                lsb['prompt_id'] = p
                system, prompt, _ = build_prompt(line, task, p)
                if 'gold' in prompt_type:
                    system = f'Given that "{line["term"]}" means "{line["meaning"]}". ' + system
                    
                if 'few' in prompt_type:
                    tmp = [system]
                    shots = get_few_shot(task)
                    for sh in shots:
                        _, prompt_sh, assistant_sh = build_prompt(sh, task, p)
                        tmp += [prompt_sh, assistant_sh]
                    tmp.append(prompt)
                    prompt = tmp
                else:
                    prompt = [system, prompt]
                
                lsb['response'] = get_response_open_source(prompt, model_name, model, tokenizer)
                w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")

def test_api(task, prompt_type, model_name, year):
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    chat = MultiChat(config,
        save_path=f"newterm/results_{year}/{task}_{model_name}{prompt_type}.jsonl",
        model=model_name,
        temperature=0
    )
    chat.start()
    
    with open(f'benchmark_{year}/{task}_clean.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            for p in range(3):
                lsb = deepcopy(line)
                lsb['prompt_id'] = p
                system, prompt, _ = build_prompt(line, task, p)
                if 'gold' in prompt_type:
                    system = f'Given that "{line["term"]}" means "{line["meaning"]}". ' + system
                if 'few' in prompt_type:
                    tmp = [system, ]
                    shots = get_few_shot(task)
                    for sh in shots:
                        _, prompt_sh, assistant_sh = build_prompt(sh, task, p)
                        tmp += [prompt_sh, assistant_sh]
                    tmp.append(prompt)
                    prompt = tmp
                else:
                    prompt = [system, prompt]
                
                role = ['user', 'assistant']
                messages = [{"role": "system", "content": prompt[0]}]
                for cnt, i in enumerate(prompt[1: ]):
                    messages.append({"role": role[cnt % 2], "content": i})
                lsb['prompt'] = messages
                chat.post(lsb)
    chat.wait_finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['COMA', 'COST', 'CSJ', 'ALL'], default='ALL', help='Which task to test.')
    parser.add_argument('--prompt', type=str, choices=['BASE', 'GOLD', 'FEW_SHOT', 'FEW_SHOT_GOLD'], default='BASE', help='Which prompt method to use.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613', help='Which model to test.')
    parser.add_argument('--year', type=str, default='2022', help='Which year to test.')

    args = parser.parse_args()
    if not os.path.exists(f"newterm/results_{args.year}"):
        os.makedirs(f"newterm/results_{args.year}")

    prompt_map = {"BASE": "", "GOLD": "_gold", "FEW_SHOT": "_few", "FEW_SHOT_GOLD": "_few_gold"}
    prompt_type = prompt_map[args.prompt]
    task_func = test_open_source
    if 'gpt-3.5' in args.model or 'gpt-4' in args.model or 'claude' in args.model:
        task_func = test_api
    
    if args.task == 'ALL':
        for task in ['COMA', 'COST', 'CSJ']:
            task_func(task, prompt_type, args.model, args.year)
    else:
        task_func(args.task, prompt_type, args.model, args.year)
