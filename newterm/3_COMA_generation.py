import re
import json
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
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").cuda()

    for dataset in "cause", "effect":
        if dataset == 'cause':
            sp = "This happened because"
            sp_full = "This happened because :"
        elif dataset == 'effect':
            sp = "As an effect"
            sp_full = "As an effect ,"

        def retry_func(input, response):
            if 'I apologize' in response:
                return False, None
            clean = []
            for sen in response.split("\n"):
                if replace(sen, input["term"], '_') is None:
                    continue
                if sp not in sen or sp.lower() + ' of' in sen.lower():
                    continue
                if len(sen.strip()):
                    clean.append(sen.strip())
            if len(clean) >= MAXIMUM:
                return False, None
            mem = "\n".join(clean) + "\n"
            return True, mem
        chat = MultiChat(config,
            retry_func=retry_func,
            save_path=f"buffer_{config["year"]}/COMA_{dataset}_generation.jsonl",
            model=config["model"],
            temperature=0.8,
            presence_penalty=0.8
        )
        chat.start()
        with open(f"benchmark_{config["year"]}/term_generation.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                lsb = {
                    "term": line["term"],
                    "meaning": line["meaning"],
                    "phrases": line["phrases"],
                    "prompt": [
                        {
                            "role": "system",
                            "content": f'Please generate {MAXIMUM + 1} different paragraphs about the new term, without using the words used above. ' + 
                                        f'Make sure that all the Sentence 1 you generate has different subject. Please print the sentence without explanation.',
                        },
                        {
                            "role": "user",
                            "content": f'I have created a new term, "{line["term"]}", which means "{line["meaning"]}". Please generate {MAXIMUM + 1} different paragraphs about "{line["term"]}", each in a separate line, following the format: "Sentence 1. {sp_full} Sentence 2."\n' + 
                                        f'Sentence 1 should contain "{line["term"]}" once. Ensure that it is objective and impartial, focusing on actual actions or events, without any emotional or subjective assumptions. ' + 
                                        f'Sentence 2, illustrating the {dataset} of Sentence 1, should be specific to "{line["term"]}" in Sentence 1 and not applicable if "{line["phrases"][0]}", "{line["phrases"][1]}", "{line["phrases"][2]}", "{line["phrases"][3]}", or "{line["phrases"][4]}" is used instead.',
                        },
                    ],
                }
                chat.post(lsb)
        chat.wait_finish()

        chat = MultiChat(config,
            save_path=f"buffer_{config["year"]}/COMA_{dataset}_generation_choice.jsonl",
            model=config["model"],
            temperature=0.8,
            presence_penalty=0.8
        )
        chat.start()
        with open(f"buffer_{config["year"]}/COMA_{dataset}_generation.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                term = line["term"]
                clean = []
                for sen in line["response"].split("\n"):
                    if replace(sen, term, '_') is None:
                        continue
                    if sp not in sen or sp.lower() + ' of' in sen.lower():
                        continue
                    sen = re.sub(r"^\d+(\.\d*)*", "", sen).strip()
                    if len(sen.strip()):
                        clean.append(sen.strip())
                clean = filter_sim(list(set(clean)), MAXIMUM, model)

                for sent in clean:
                    questions = sent.split(sp)
                    assert len(questions) == 2
                    question = questions[0].strip()
                    gold = questions[1].strip(':').strip(',').strip()
                    tmp_len = len(gold.split(" "))
                    for newterm in line["phrases"]:
                        tmp = replace(question, term, newterm)
                        lsb = {
                            "term": term,
                            "meaning": line["meaning"],
                            "question": question,
                            "gold": gold,
                            "choice": newterm,
                            "prompt": [
                                {
                                    "role": "system",
                                    "content": f"Please generate one sentence with {tmp_len} words to finish the following paragraph. Please print the sentence without explanation.",
                                },
                                {
                                    "role": "user", 
                                    "content": f"{tmp} {sp_full} "
                                },
                            ],
                        }
                        chat.post(lsb)
        chat.wait_finish()

        chat = MultiChat(config,
            save_path=f"buffer_{config["year"]}/COMA_{dataset}_generation_filter.jsonl",
            model=config["model"],
            temperature=0
        )
        chat.start()
        mem = defaultdict(list)
        with open(f"buffer_{config["year"]}/COMA_{dataset}_generation_choice.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                if line["choice"].lower() not in line["response"].lower():
                    mem[(line["term"], line["meaning"], line["question"], line["gold"])].append(line["response"])

        for k, v in mem.items():
            k = list(k)
            term = k[0]
            meaning = k[1]
            question = k[2].strip().strip(':').strip().strip(',').strip()
            k[3] = k[3].strip().strip(':').strip().strip(',').strip()
            v = [i.strip().strip(':').strip().strip(',').strip() for i in v]
            choices = [k[3]] + v
            if len(choices) < 6:
                continue
            choices = filter_sim(choices, 6, model, skip_id=[0])
            random.seed(len(choices[0]))
            random.shuffle(choices)
            answer = choices.index(k[3])
            try:
                lsb = {
                    "term": term,
                    "meaning": meaning,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "prompt": [
                        {
                            "role": "system",
                            "content": f'Given that "{term}" means "{meaning}", please answer the following question by selecting the most probable options. If multiple choices have equal likelihood, you may choose more than one. List the selected choices (A, B, C, D, E or F) separated by commas.',
                        },
                        {
                            "role": "user",
                            "content": f'Please solve the following multiple choice exercise:\nExercise: choose the most plausible alternative.\n\n{question} {"because" if dataset == "cause" else "so"}...\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nE. {choices[4]}\nF. {choices[5]}\nAnswer: ',
                        },
                    ],
                }
                chat.post(lsb)
            except:
                continue
        chat.wait_finish()

    by_word = defaultdict(list)
    for dataset in "cause", "effect":
        if dataset == 'cause':
            sp = "This happened because"
            sp_full = "This happened because :"
        elif dataset == 'effect':
            sp = "As an effect"
            sp_full = "As an effect ,"

        with open(f"buffer_{config["year"]}/COMA_{dataset}_generation_filter.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                mem = ["A", "B", "C", "D", "E", "F"][: len(line["choices"])]
                asw = mem[line["answer"]]
                tmp = [i.split('.')[0].strip() for i in line["response"].split(",")]
                for i in tmp:
                    if i in mem:
                        mem.remove(i)
                if asw in mem or len(mem) < 3:
                    continue
                mem.append(asw)

                choices = []
                for i in mem:
                    choices.append(line["choices"][CHOICE.index(i)])
                choices = filter_sim(choices, 4, model, skip_sent=[line["choices"][line["answer"]]])
                tot = 0
                for i in choices:
                    tot += i.count('"')
                if tot >= 4:
                    continue
                random.seed(len(line['question']))
                random.shuffle(choices)
                asw = choices.index(line["choices"][line["answer"]])
                if replace(line['question'], line['term'], '') is None and sum([replace(i, line['term'], '') is not None for i in line['choices']]) == 0:
                    continue
                by_word[(line['term'], line['meaning'])].append([line['question'], choices, asw, sp])

    to_type = {}
    with open(f'benchmark_{config["year"]}/new_terms.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            to_type[(line['term'], line['meaning'])] = line['type']

    login(token=config["huggingface_key"])
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").half().cuda()

    def get_final(by_word, instance_per_word):
        by_word = deepcopy(by_word)
        for k, v in by_word.items():
            scores = []
            for it in v:
                sent = it[0] + ' ' + it[-1] + ' ' + it[1][it[2]]
                score = sen_ppl(sent, tokenizer, model, it[-1])
                scores.append(score)
            sele = sorted([(i, j) for i, j in zip(scores, v)])
            by_word[k] = sele[-instance_per_word * 2: ]
        final = defaultdict(list)
        for k in by_word.keys():
            cnt = instance_per_word
            while len(by_word[k]) and cnt:
                it = by_word[k][0][1]
                final[to_type[k]].append({"term": k[0], "meaning": k[1], "question": it[0], "gold": it[2], "choices": it[1], "split": "cause" if "cause" in it[-1] else "effect"})
                by_word[k].pop(0)
                cnt -= 1

        num = config["term_per_category"] * instance_per_word
        tmp = defaultdict(list)
        for k, v in by_word.items():
            for it in v:
                dct = {"term": k[0], "meaning": k[1], "question": it[1][0], "gold": it[1][2], "choices": it[1][1], "split": "cause" if "cause" in it[1][-1] else "effect"}
                tmp[to_type[k]].append((it[0], dct))
        for k, v in tmp.items():
            v = sorted(v, key=lambda x: x[0])
            rest = num - len(final[k])
            for i, j in v[-rest: ]:
                final[k].append(j)

        return final

    with open(f'benchmark_{config["year"]}/COMA.jsonl', 'w', encoding='utf-8') as f:
        for k, v in get_final(by_word, config["instance_per_word"]).items():
            for it in v:
                it['type'] = k
                f.write(json.dumps(it, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
