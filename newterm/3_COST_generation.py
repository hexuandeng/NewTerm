import re
import json
import math
import random
from copy import deepcopy
from huggingface_hub import login
from collections import defaultdict
from transformers import AutoTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from utils import MultiChat, sen_ppl, replace, filter_sim

if __name__ == "__main__":
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    CHOICE = ["A", "B", "C", "D", "E", "F"]
    MAXIMUM = config["instance_per_word"] * 2
    model = SentenceTransformer('whaleloops/phrase-bert').cuda()
    
    def retry_func(input, response):
        if 'I apologize' in response:
            return False, None
        clean = []
        for sen in response.split("\n"):
            if replace(sen, input["term"], '_') is None:
                continue
            if len(sen.strip()):
                clean.append(sen.strip())
        if len(clean) >= MAXIMUM:
            return False, None
        mem = "\n".join(clean) + "\n"
        return True, mem
    chat = MultiChat(config,
        retry_func=retry_func,
        save_path=f"buffer_{config["year"]}/COST_generation.json",
        model=config["model"],
        temperature=0.8,
        presence_penalty=0.8
    )
    chat.start()
    with open(f"benchmark_{config["year"]}/term_generation.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            phrases = filter_sim(line["phrases"].copy(), 5, model)
            lsb = {
                "term": line["term"],
                "meaning": line["meaning"],
                "phrases": line["phrases"],
                "prompt": [
                    {
                        "role": "system",
                        "content": f'Please generate {MAXIMUM + 1} different sentences about the new term, each in a separate line, without using the words used above. ' + 
                                    f'Make sure that all the sentence you generate has different subject. Please print the sentence without explanation.',
                    },
                    {
                        "role": "user",
                        "content": f'I have created a new term, "{line["term"]}", which means "{line["meaning"]}". Please generate {MAXIMUM + 1} different sentences about "{line["term"]}", each in a separate line, ' + 
                                    f'which should be specific to "{line["term"]}". The sentence should be grammatically correct but not applicable if "{phrases[0]}", "{phrases[1]}", "{phrases[2]}", "{phrases[3]}", or "{phrases[4]}" is used instead.',
                    },
                ],
            }
            chat.post(lsb)
    chat.wait_finish()

    chat = MultiChat(config,
        save_path=f"buffer_{config["year"]}/COST_generation.json",
        model=config["model"],
        temperature=0.8,
        presence_penalty=0.8
    )
    chat.start()
    with open(f"benchmark_{config["year"]}/term_generation.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            term = line["term"]
            meaning = line["meaning"]
            phrases = filter_sim(line["phrases"].copy(), MAXIMUM, model)
            num = math.ceil(MAXIMUM / len(phrases))
            for newterm in phrases:
                phs = filter_sim(line["phrases"].copy(), 5, model, skip_sent=[newterm])
                phs.remove(newterm)
                lsb = {
                    "term": term,
                    "meaning": meaning,
                    "phrases": line["phrases"],
                    "gold": newterm,
                    "prompt": [
                        {
                            "role": "system",
                            "content": f'Please generate {num} sentence that "{newterm}" is exactly in the sentence but not its other forms. Please print the sentence without explanation.'
                        },
                        {
                            "role": "user", 
                            "content": f'Please generate {num} sentence about "{newterm}", which should be specific to "{newterm}". The sentence should be grammatically correct but not applicable if "{phs[0]}", "{phs[1]}", "{phs[2]}", or "{phs[3]}" is used instead.\nSentence: '
                        },
                    ],
                }
                chat.post(lsb)
    chat.wait_finish()

    chat = MultiChat(config,
        save_path=f"buffer_{config["year"]}/COST_generation_filter.json",
        model=config["model"],
        temperature=0
    )
    chat.start()
    mem = defaultdict(list)
    with open(f"buffer_{config["year"]}/COST_generation.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            term = line['term']
            meaning = line['meaning']
            choices = line['phrases']
            if 'gold' in line.keys():
                gold = line['gold']
                question = [re.sub(r"^\d+(\.\d*)*", "", line['response']).strip()]
            else:
                gold = term
                question = [re.sub(r"^\d+(\.\d*)*", "", i).strip() for i in line['response'].split('\n')]
            question = [replace(i, gold, '_') for i in question]
            question = [re.sub(r"_[a-zA-Z]*", "_", i) for i in question if i is not None and i.count('_') == 1]
            if question is None:
                continue

            choices += [term]
            choices = filter_sim(choices, 6, model, skip_sent=[gold, term])
            choices = [i[0].upper() + i[1: ] for i in choices]
            random.seed(len(meaning))
            random.shuffle(choices)
            asw = choices.index(gold[0].upper() + gold[1: ])
            for q in question:
                lsb = {
                    "term": term,
                    "meaning": meaning,
                    "question": q,
                    "answer": asw,
                    "choices": choices,
                    "prompt": [
                        {
                            "role": "system",
                            "content": f'Please answer the following choice question by selecting the most probable options. If multiple choices have equal likelihood, you may choose more than one. List the selected choices (A, B, C, D, E or F) separated by commas.',
                        },
                        {
                            "role": "user",
                            "content": f'Given that the term "{term}" means "{meaning}", Please solve the following multiple choice exercise:\n"{q}"\nReplace the _ in the above sentence with the correct option:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nF. {choices[5]}\nAnswer: ',
                        },
                    ],
                }
                chat.post(lsb)
    chat.wait_finish()

    by_word = defaultdict(list)
    with open(f"buffer_{config["year"]}/COST_generation_filter.json", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            for cnt, i in enumerate(line["choices"]):
                line["choices"][cnt] = i.strip().strip(':').strip().strip(',').strip()
            mem = ["A", "B", "C", "D", "E", "F"]
            asw = mem[line["answer"]]
            try:
                new = mem[line["choices"].index(line["term"][0].upper() + line["term"][1: ])]
            except:
                continue
            tmp = [i.split('.')[0].strip() for i in line["response"].split(",")]
            for i in tmp:
                if i in mem:
                    mem.remove(i)
            if asw in mem or len(mem) < 3:
                continue
            mem.append(asw)
            if new not in mem:
                continue

            choices = []
            for i in mem:
                choices.append(line["choices"][CHOICE.index(i)])
            choices = filter_sim(choices, 4, model, skip_sent=[line["choices"][line["answer"]], line["term"][0].upper() + line["term"][1: ]])
            random.seed(len(line['question']))
            random.shuffle(choices)
            asw = choices.index(line["choices"][line["answer"]])
            if 'Here are 3' in line['question']:
                continue
            if len([i for i in line['question'].replace('_', '').split() if len(i.strip())]) < 3:
                continue
            by_word[(line['term'], line['meaning'])].append([line['question'], choices, asw])

    login(token=config["huggingface_key"])
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").half().cuda()

    to_type = {}
    with open(f'benchmark_{config["year"]}/new_terms.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            to_type[(line['term'], line['meaning'])] = line['type']

    def get_final(by_word, instance_per_word):
        by_word = deepcopy(by_word)
        for k, v in by_word.items():
            scores = []
            for it in v:
                sent = it[0].replace('_', it[1][it[2]].replace('’', "'"))
                score = sen_ppl(sent, tokenizer, model)
                scores.append(score)
            sele1 = sorted([(i, j) for i, j in zip(scores, v) if j[1][j[2]].lower() == k[0].lower()])
            sele2 = sorted([(i, j) for i, j in zip(scores, v) if j[1][j[2]].lower() != k[0].lower()])
            by_word[k] = []
            if len(sele1):
                by_word[k] += sele1[-instance_per_word: ]
            if len(sele2) >= instance_per_word or (len(sele1) and len(sele2) and sele2[-1][0] >= sele1[-1][0]):
                by_word[k] += sele2[-instance_per_word: ]
            random.seed(len(k[1]))
            random.shuffle(by_word[k])
        final = defaultdict(list)
        for k in by_word.keys():
            cnt = instance_per_word
            while len(by_word[k]) and cnt:
                it = by_word[k][0][1]
                final[to_type[k]].append({"term": k[0], "meaning": k[1], "question": it[0], "gold": it[2], "choices": it[1]})
                by_word[k].pop(0)
                cnt -= 1

        num = config["term_per_category"] * instance_per_word
        tmp = defaultdict(list)
        for k, v in by_word.items():
            for it in v:
                dct = {"term": k[0], "meaning": k[1], "question": it[1][0], "gold": it[1][2], "choices": it[1][1]}
                tmp[to_type[k]].append((it[0], dct))
        for k, v in tmp.items():
            v = sorted(v, key=lambda x: x[0])
            rest = num - len(final[k])
            if rest:
                for i, j in v[-rest: ]:
                    final[k].append(j)

        return final

    with open(f'benchmark_{config["year"]}/COST.json', 'w', encoding='utf-8') as f:
        for k, v in get_final(by_word, config["instance_per_word"]).items():
            for it in v:
                it['type'] = k
                f.write(json.dumps(it, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
