import os
import json
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from collections import defaultdict
from utils import MultiChat
from loguru import logger

if __name__ == "__main__":
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    chat = MultiChat(config,
        save_path=f'buffer_{config["year"]}/new_term_guess.json',
        model=config["model"],
        temperature=0
    )
    chat.start()
    with open(f'buffer_{config["year"]}/new_terms_raw.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            line = line.strip().split('\t')
            if '(' in line[0] and ')' in line[0]:
                if line[0].strip()[-1] != ')':
                    # Filter out complex phrases
                    continue
                line[0] = line[0].split('(')[0].strip()
            lsb = {
                'term': line[0],
                'meaning': line[1].strip(),
                'prompt': [{"role": "system", "content": 'Please deduce the meaning of the following word based on its spelling, using just one sentence.'}, 
                        {"role": "user", "content": f'What is the meaning of "{line[0]}"?\nMeaning: '}]
            }
            chat.post(lsb)

    # get the API KEY here: https://developers.google.com/custom-search/v1/overview
    # get your Search Engine ID on your CSE control panel: https://cse.google.com/cse/
    model = SentenceTransformer('whaleloops/phrase-bert').cuda()
    used = []
    cnt = 0
    if os.path.exists(f'buffer_{config["year"]}/new_term_google.json'):
        with open(f'buffer_{config["year"]}/new_term_google.json', 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                del line['count']
                used.append(line)
    chat.wait_finish()
    logger.info('Starting generating and writing to buffer/new_term_google.json!')
    with open(f'buffer_{config["year"]}/new_term_guess.json', 'r', encoding='utf-8') as f,\
        open(f'buffer_{config["year"]}/new_term_google.json', 'a', encoding='utf-8') as w:
        for line in f:
            line = json.loads(line)
            if line in used:
                continue
            request = f'["{line["term"]}" before:{config["cutoff_date"]}]'
            url = f"https://www.googleapis.com/customsearch/v1?key={config['google_custom_search_API']}&cx={config['google_search_engine_ID']}&q={request}&alt=json&fields=queries(request(totalResults))"
            data = requests.get(url).json()
            if data['queries']['request'][0] == {}:
                line['count'] = 0
            else:
                line['count'] = int(data['queries']['request'][0]['totalResults'])
            w.write(json.dumps(line, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
            cnt += 1
    logger.info(f'Finish generating {cnt} instances in buffer/new_term_google.json!')

    all_score = defaultdict(list)
    mem = {}
    with open(f'buffer_{config["year"]}/new_term_google.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            asw = cos_sim(model.encode(line['response']), model.encode(line['meaning']))
            all_score[('guess', ' ' in line['term'])].append(asw[0][0].item())
            if line['count'] > 10:
                all_score[('google', ' ' in line['term'])].append(line['count'])
            mem[(line['term'], line['meaning'])] = (asw[0][0].item(), line['count'])
    sep = {
        ('guess', True): [
            all_score[('guess', True)][int(len(all_score[('guess', True)]) / 3)],
            all_score[('guess', True)][int(len(all_score[('guess', True)]) / 3 * 2)]
            ],
        ('guess', False): [
            all_score[('guess', False)][int(len(all_score[('guess', False)]) / 3)],
            all_score[('guess', False)][int(len(all_score[('guess', False)]) / 3 * 2)]
            ],
        ('google', True): [10, all_score[('google', True)][int(len(all_score[('google', True)]) / 2)]],
        ('google', False): [10, all_score[('google', False)][int(len(all_score[('google', False)]) / 2)]]
    }
    with open(f'benchmark_{config["year"]}/new_terms.txt', 'w', encoding='utf-8') as f:
        for cat in config['selected_type']:
            logger.info(f'Selecting {cat}!')
            if 'word' in cat:
                typ = False
                logger.info(f'Total {len(all_score[("guess", typ)])} words!')
            elif 'phrase' in cat:
                typ = True
                logger.info(f'Total {len(all_score[("guess", typ)])} phrases!')
            else:
                raise NotImplementedError
            
            # select at least twice and at most five times as large as term_per_category
            if 'new' in cat:
                start = 0
                end = 10
            elif 'rare' in cat:
                start = 10
                end = min(max(sep[('google', typ)][1], 
                sorted(all_score[('google', typ)])[config['term_per_category'] * 2]),
                sorted(all_score[('google', typ)])[config['term_per_category'] * 5])
            elif 'freque' in cat:
                start = max(min(sep[('google', typ)][1], 
                sorted(all_score[('google', typ)])[-config['term_per_category'] * 2]),
                sorted(all_score[('google', typ)])[-config['term_per_category'] * 5])
                end = float('inf')
            all_sele = []
            for k, v in mem.items():
                if v[1] >= start and v[1] <= end and (' ' in k[0]) == typ:
                    all_sele.append((v[0], k))
            logger.info(f'Selecting {len(all_sele)} by frequency!')
            all_sele = sorted(all_sele)
            percent = config['term_per_category'] / len(all_sele)
            if 'not deduced' in cat:
                start = 0.0
                end = percent
            elif 'partially deduced' in cat:
                start = (1 - percent) / 2
                end = (1 + percent) / 2
            elif 'fully deduced' in cat:
                start = 1 - percent
                end =  1
            all_sele = all_sele[int(len(all_sele) * start): int(len(all_sele) * end)]
            logger.info(f'Selecting {len(all_sele)} {cat} by deduced difficulties!')
            for i in all_sele:
                f.write(json.dumps({
                    'term': i[1][0],
                    'meaning': i[1][1],
                    'type': cat
                }, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
