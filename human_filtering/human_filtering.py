import re
import json
import glob
from collections import defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa

YEAR = 2023

if __name__ == "__main__":
    with open(f'human_filtering/questions_{YEAR}.json', 'r', encoding='utf-8') as f:
        all = json.load(f)
    history = {}
    for i in all:
        history[i['name']] = i

    count = defaultdict(lambda: [0, 0])
    table = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    multi = defaultdict(int)
    none = defaultdict(int)
    error = defaultdict(int)
    files = glob.glob(f'benchmark_{YEAR}/filtering/*.json')
    to_del = []
    for p in files:
        with open(p, 'r', encoding='utf8') as f:
            obj = json.load(f)
            for it, value in obj.items():
                match = re.match(r"([a-z]+)([0-9]+)", it, re.I)
                items = match.groups()[0]
                if '-Comment' in it:
                    continue
                if value == 'none':
                    table[it][-1] += 1
                elif value not in history[it]["choices"]:
                    table[it][-2] += 1
                else:
                    table[it][history[it]["choices"].index(value)] += 1

                count[items][0] += int(history[it]['correctAnswer'] == value)
                count[items][1] += 1
                if history[it]['correctAnswer'] != value:
                    to_del.append(history[it]['title'])
                    if value == 'none':
                        none[items] += 1
                    elif value not in history[it]["choices"]:
                        multi[items] += 1
                    else:
                        error[items] += 1

    tables = []
    count2 = 0
    count3 = 0
    for k, v in table.items():
        if 2 in v:
            count2 += 1
        elif 3 in v:
            count3 += 1
        tables.append(v)
    print(f"Fleiss Kappa: {fleiss_kappa(tables)}")
    print(count2, count3)

    a = b = 0
    for k in count.keys():
        print('Dataset:', k)
        print('Correct Rate:', count[k][0] / count[k][1])
        print('Multi.:', multi[k], ', None:', none[k], ', Wrong:', error[k])
        a += count[k][0]
        b += count[k][1]
        print()
    print('Total Correct Rate:', a / b)

    for data in ['COMA', 'CSJ']:
        with open(f"benchmark_{YEAR}/{data}.jsonl", 'r', encoding='utf-8') as f,\
            open(f"benchmark_{YEAR}/{data}_clean.jsonl", 'w', encoding='utf-8') as w:
            for cnt, line in enumerate(f):
                js = json.loads(line)
                flag = False
                for k in to_del:
                    if js['term'] in k and js['meaning'] in k and js['question'] in k:
                        flag = True 
                        break
                if not flag:
                    w.write(line)
    with open(f"benchmark_{YEAR}/COST.jsonl", 'r', encoding='utf-8') as f,\
        open(f"benchmark_{YEAR}/COST_clean.jsonl", 'w', encoding='utf-8') as w:
        for cnt, line in enumerate(f):
            js = json.loads(line)
            flag = False
            for k in to_del:
                if js['term'] in k and js['meaning'] in k and js['question'].replace('_', '<u>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;</u>') in k:
                    flag = True 
                    break
            if not flag:
                w.write(line)
