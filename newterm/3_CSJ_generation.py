import re
import json
from copy import deepcopy
from huggingface_hub import login
from collections import defaultdict
from transformers import AutoTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from utils import MultiChat, sen_ppl, replace, filter_sim

if __name__ == "__main__":
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
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
        save_path=f"buffer_{config["year"]}/CSJ_generation.jsonl",
        model=config["model"],
        temperature=0.8,
        presence_penalty=0.8
    )
    chat.start()
    with open(f"benchmark_{config["year"]}/term_generation.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            line["phrases"] = filter_sim(line["phrases"], 5, model)
            lsb = {
                "term": line["term"],
                "meaning": line["meaning"],
                "phrases": line["phrases"],
                "answer": True,
                "prompt": [
                    {
                        "role": "system",
                        "content": f'Please generate {MAXIMUM + 1} different sentences about the new term, each in a separate line, without using the words used above. ' + 
                                    f'Make sure that all the sentence you generate has different subject. Please print the sentence without explanation.',
                    },
                    {
                        "role": "user",
                        "content": f'I have created a new term, "{line["term"]}", which means "{line["meaning"]}". Please generate {MAXIMUM + 1} different sentences about "{line["term"]}", each in a separate line, ' + 
                                    f'which should be specific to "{line["term"]}". The sentence should be grammatically correct but not applicable if "{line["phrases"][0]}", "{line["phrases"][1]}", "{line["phrases"][2]}", "{line["phrases"][3]}", or "{line["phrases"][4]}" is used instead.',
                    },
                ],
            }
            chat.post(lsb)
    chat.wait_finish()

    def retry_func(input, response):
        if 'I apologize' in response:
            return False, None
        clean = []
        flag = False
        for sen in response.split("\n"):
            if flag:
                it = sen
                sen = "Wrong Sentence: " + sen
            else:
                if 'wrong sentence' not in sen.lower():
                    continue
                it = re.split("Wrong Sentence", sen, flags=re.IGNORECASE)[-1]
                cnt = 0
                for i in it:
                    if i.isalpha():
                        cnt += 1
                if cnt == 0:
                    flag = True
                    continue
            if replace(it, input["term"], '_') is None:
                continue
            if len(sen.strip()):
                clean.append(sen.strip())
        if len(clean) >= MAXIMUM:
            return False, None
        mem = "\n".join(clean) + "\n"
        return True, mem
    chat = MultiChat(config,
        retry_func=retry_func,
        save_path=f"buffer_{config["year"]}/CSJ_generation.jsonl",
        model=config["model"],
        temperature=0.8,
        presence_penalty=0.8
    )
    chat.start()
    with open(f"buffer_{config["year"]}/CSJ_generation.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if not line['answer']:
                continue
            lsb = {
                "term": line["term"],
                "meaning": line["meaning"],
                "answer": False,
                "prompt": [
                    {
                        "role": "system",
                        "content": f'Please generate {MAXIMUM + 1} different sentences about the new term, each in a separate line, without using the words used above. ' + 
                                    f'Make sure that all the sentence you generate has different subject.',
                    },
                    {
                        "role": "user",
                        "content": f'I have created a new term, "{line["term"]}", which means "{line["meaning"]}". Please generate {MAXIMUM + 1} different sentences about "{line["term"]}", each in a separate line, ' + 
                                    f'which should be specific to "{line["term"]}". The sentence should be grammatically correct but not applicable if "{line["phrases"][0]}", "{line["phrases"][1]}", "{line["phrases"][2]}", "{line["phrases"][3]}", or "{line["phrases"][4]}" is used instead.',
                    },
                    {
                        "role": "assistant",
                        "content": line["response"],
                    },
                    {
                        "role": "user",
                        "content": f'For each sentence generated above, please modify it to use "{line["term"]}" illogically, based on the given meaning, while keeping the grammar, fluency, and original subject intact. ' +
                                    'For each example, print "Wrong Sentence:" and "Corresponding Wrong meaning:" on separate lines, explaining the deviation from the intended meaning. Ensure that each wrong meaning is significantly different from those previously generated.',
                    }
                ],
            }
            chat.post(lsb)
    chat.wait_finish()

    chat = MultiChat(config,
        save_path=f"buffer_{config["year"]}/CSJ_generation_filter.jsonl",
        model=config["model"],
        temperature=0
    )
    chat.start()
    with open(f"buffer_{config["year"]}/CSJ_generation.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            questions = []
            for sent in line['response'].split('\n'):
                if not line['answer']:
                    if 'wrong sentence' not in sent.lower():
                        continue
                    sent = re.split("Corresponding Wrong meaning", sent, flags=re.IGNORECASE)[0]
                    sent = re.split("Wrong Sentence", sent, flags=re.IGNORECASE)[-1]
                sent = sent.strip().strip(':').strip().strip(',').strip()
                sent = re.sub(r"^\d+(\.\d*)*", "", sent).strip()
                questions.append(sent)

            for question in questions:
                lsb = {
                    "term": line["term"],
                    "meaning": line["meaning"],
                    "question": question,
                    "answer": line["answer"],
                    "prompt": [
                        {
                            "role": "system",
                            "content": f'Please answer the following question with an integer, without any further explination.',
                        },
                        {
                            "role": "user",
                            "content": f'Given that "{line["term"]}" means "{line["meaning"]}". On a scale of 0 to 10, with 0 being extremely unlikely and 10 being highly likely, how probable is it that the following sentence coherent and align with general understanding?\n{question}\nAnswer: ',
                        },
                    ],
                }
                chat.post(lsb)
    chat.wait_finish()

    by_word = defaultdict(list)
    with open(f"buffer_{config["year"]}/CSJ_generation_filter.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            try:
                score = int(line['response'])
            except:
                continue
            if line['answer'] and score <= 7:
                continue
            if (not line['answer']) and score > 3:
                continue
            if replace(line['question'], line['term'], '') is None:
                continue
            by_word[(line['term'], line['meaning'])].append([line['question'], line['answer']])
                
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
                sent = it[0].replace('’', "'")
                score = sen_ppl(sent, tokenizer, model)
                scores.append(score)
            sele = sorted([(i, j) for i, j in zip(scores, v)])
            by_word[k] = sele[-instance_per_word * 2: ]
        final = defaultdict(list)
        for k in by_word.keys():
            cnt = instance_per_word
            while len(by_word[k]) and cnt:
                it = by_word[k][0][1]
                final[to_type[k]].append({"term": k[0], "meaning": k[1], "question": it[0], "gold": it[1]})
                by_word[k].pop(0)
                cnt -= 1

        num = config["term_per_category"] * instance_per_word
        tmp = defaultdict(list)
        for k, v in by_word.items():
            for it in v:
                dct = {"term": k[0], "meaning": k[1], "question": it[1][0], "gold": it[1][1]}
                tmp[to_type[k]].append((it[0], dct))
        for k, v in tmp.items():
            v = sorted(v, key=lambda x: x[0])
            rest = num - len(final[k])
            if rest:
                for i, j in v[-rest: ]:
                    final[k].append(j)

        return final

    with open(f'benchmark_{config["year"]}/CSJ.jsonl', 'w', encoding='utf-8') as f:
        for k, v in get_final(by_word, config["instance_per_word"]).items():
            for it in v:
                it['type'] = k
                f.write(json.dumps(it, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
