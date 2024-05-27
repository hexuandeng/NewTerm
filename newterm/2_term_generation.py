import re
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from utils import MultiChat, filter_sim

if __name__ == "__main__":
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    chat = MultiChat(config, 
        save_path=f'buffer_{config["year"]}/word_generation_sim.json',
        model=config["model"],
        temperature=0
    )
    chat.start()
    with open(f'benchmark_{config["year"]}/new_terms.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            lsb = {
                'term': line["term"],
                'meaning': line["meaning"],
                'prompt': [{"role": "system", "content": 'Please provide three words and three two-word phrases, and display each of them on a separate line. The first three lines are words, each on a separate line, and the last three lines are phrases, each on a separate line. Make sure that there have six lines in total, with each word/phrase at a single line.'}, 
                        {"role": "user", "content": f'Please provide three words and three phrases, capturing a different aspect of the following concept: "{line["meaning"]}". Ensure that these are commonly used and easily understood by a 3-year-old child.'}]
            }
            chat.post(lsb)
    chat.wait_finish()

    chat = MultiChat(config,
        save_path=f'buffer_{config["year"]}/word_generation_synonym.json',
        model=config["model"],
        temperature=0
    )
    chat.start()
    with open(f'benchmark_{config["year"]}/new_terms.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            lsb = {
                'term': line["term"],
                'meaning': line["meaning"],
                'prompt': [{"role": "system", "content": 'Please answer the following question by printing three terms without explanation, each at a seperate line. If you cannot construct a sentence that fully meets the requirements, provide words that partially fulfill the requirements. Do not refrain from answering.'}, 
                        {"role": "user", "content": f'What is the synonym for the new term "{line["term"]}", that refers to "{line["meaning"]}"? The synonym should be a commonly used English term and belong to the same part of speech. Do not use abbreviations and comma, period in the term, and shorter than five words. Please generate three different alternatives.\nSynonym:\n'}]
            }
            chat.post(lsb)
    chat.wait_finish()

    chat = MultiChat(config,
        save_path=f'buffer_{config["year"]}/word_generation_antonym.json',
        model=config["model"],
        temperature=0
    )
    chat.start()
    with open(f'benchmark_{config["year"]}/new_terms.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            lsb = {
                'term': line["term"],
                'meaning': line["meaning"],
                'prompt': [{"role": "system", "content": 'Please answer the following question by printing three terms without explanation, each at a seperate line. If you cannot construct a sentence that fully meets the requirements, provide words that partially fulfill the requirements. Do not refrain from answering.'}, 
                        {"role": "user", "content": f'What is the antonym for the new term "{line["term"]}", that refers to "{line["meaning"]}"? The antonym should be a commonly used English term and have a completely different meaning but belong to the same part of speech. Do not use abbreviations and comma, period in the term, and shorter than five words. Please generate three different alternatives.\nAntonym:\n'}]
            }
            chat.post(lsb)
    chat.wait_finish()

    chat = MultiChat(config,
        save_path=f'buffer_{config["year"]}/word_generation_guess.json',
        model=config["model_guess"],
        temperature=0
    )
    chat.start()
    with open(f'benchmark_{config["year"]}/new_terms.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            lsb = {
                'term': line["term"],
                'meaning': line["meaning"],
                'prompt': [{"role": "system", "content": 'Please answer the following question by printing three terms without explanation, each at a seperate line. If you cannot construct a sentence that fully meets the requirements, provide words that partially fulfill the requirements. Do not refrain from answering.'}, 
                        {"role": "user", "content": f'Please guess the meaning of the term "{line["term"]}" and create three alternative terms based on its spelling. Alternative term:\n'}]
            }
            chat.post(lsb)
    chat.wait_finish()

    word_mem = defaultdict(list)
    for file in ['sim', 'synonym', 'antonym', 'guess']:
        with open(f'buffer_{config["year"]}/word_generation_{file}.json', 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                words = [re.sub(r"^\d+(\.\d*)*", "", i).strip() for i in line['response'].split('\n') if len(i.split()) <= 5]
                for i in words:
                    if (not len(i)) or ':' in i or '.' in i or 'none' in i.lower() or 'n/a' in i.lower():
                        continue
                    word_mem[(line["term"], line["meaning"])].append(i)

    model = SentenceTransformer('whaleloops/phrase-bert').cuda()
    with open(f'benchmark_{config["year"]}/term_generation.json', 'w', encoding='utf-8') as f:
        for k, v in word_mem.items():
            phrases = filter_sim([k[1]] + v, 6, model, skip_id=[0])
            phrases.pop(0)
            lsb = {
                'term': k[0],
                'meaning': k[1],
                'phrases': phrases
            }
            f.write(json.dumps(lsb, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
            word_mem[k] = phrases
