import json
import argparse
from copy import deepcopy
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
newterm_path = os.path.abspath(os.path.join(current_dir, '../../newterm'))
sys.path.insert(1, newterm_path)
from utils import build_prompt, get_response_open_source, load_open_source, MultiChat

def test_open_source(task, prompt_type, model_name, year, path):
    model, tokenizer = load_open_source(model_name)

    rag_name = os.path.join(path, f'{task}_rag.jsonl')
    
    with open(rag_name, 'r', encoding='utf-8') as f, \
         open(os.path.join(path, f'results_{year}', f'{task}_{model_name.split("/")[-1]}{prompt_type}.jsonl'), 'w', encoding='utf-8') as w:
        
        for line in f:
            line = json.loads(line.strip())
            
            # Loop through prompt variants
            for p in range(3):
                lsb = deepcopy(line)
                lsb['prompt_id'] = p
                system, prompt, _ = build_prompt(line, task, p)
                
                # Handle the different prompt types
                if 'gold' in prompt_type:
                    system = f'Given that "{line["term"]}" means "{line["meaning"]}". ' + system
                    prompt = [system, prompt]
                elif 'rag' in prompt_type:
                    tmp = f"I've come across a new term, \"{line['term']}\". To help you understand it better, I've gathered some sentences that are relevant to this term. Here they are:\n"
                    for it in line["rag"][:5]:
                        tmp += it + "\n"
                    tmp += f"\nAfter reviewing these sentences, could you please assist me in answering the following question about the new term \"{line['term']}\"? {system}\n\n"
                    prompt = [system, tmp + prompt]
                else:  # No hint case
                    prompt = [system, prompt]
                    
                lsb['response'] = get_response_open_source(prompt, model_name, model, tokenizer)
                w.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
    del model
    del tokenizer

def test_api(task, prompt_type, model_name, year, path):
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    chat = MultiChat(config,
        save_path=f"{path}results_{year}/{task}_{model_name}{prompt_type}.jsonl",
        model=model_name,
        temperature=0
    )
    chat.start()

    rag_name = f'{path}{task}_rag.jsonl' 
    # Load RAG data
    with open(rag_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            
            # Loop through prompt variants
            for p in range(3):
                lsb = deepcopy(line)
                lsb['prompt_id'] = p
                system, prompt, _ = build_prompt(line, task, p)
                
                # Handle the different prompt types
                if 'gold' in prompt_type:
                    system = f'Given that "{line["term"]}" means "{line["meaning"]}". ' + system
                    prompt = [system, prompt]
                elif '_rag' in prompt_type:
                    tmp = f"I've come across a new term, \"{line['term']}\". To help you understand it better, I've gathered some sentences that are relevant to this term. Here they are:\n"
                    for it in line["rag"][:5]:
                        tmp += it + "\n"
                    tmp += f"\nAfter reviewing these sentences, could you please assist me in answering the following question about the new term \"{line['term']}\"? {system}\n\n"
                    prompt = [system, tmp + prompt]
                else:  # No hint case
                    prompt = [system, prompt]
                
                # Construct message
                role = ['user', 'assistant']
                messages = [{"role": "system", "content": prompt[0]}]
                for cnt, i in enumerate(prompt[1:]):
                    messages.append({"role": role[cnt % 2], "content": i})
                lsb['prompt'] = messages
                chat.post(lsb)

    chat.wait_finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['COMA', 'COST', 'CSJ', 'ALL'], default='COMA', help='Which task to test.')
    parser.add_argument('--prompt', type=str, choices=['BASE', 'GOLD', 'RAG'], default='BASE', help='Which prompt method to use.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Which model to test.')
    parser.add_argument('--year', type=str, default='2023', help='Which year to test.')
    parser.add_argument('--path', type=str, default='data/', help='Base path for data files.')

    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.path, f"results_{args.year}")):
        os.makedirs(os.path.join(args.path, f"results_{args.year}"))

    prompt_map = {
        "BASE": "", 
        "GOLD": "_gold", 
        "RAG": "_rag", 
    }
    prompt_type = prompt_map[args.prompt]
    
    task_func = test_open_source if 'gpt-3.5' not in args.model and 'gpt-4' not in args.model and 'claude' not in args.model else test_api
    if args.task == 'ALL':
        for task in ['COMA', 'COST', 'CSJ']:
            task_func(task, prompt_type, args.model, args.year, args.path)
    else:
        task_func(args.task, prompt_type, args.model, args.year, args.path)
