import re
import json
import glob
import argparse
from copy import deepcopy
from collections import defaultdict
pattern = re.compile('[\W_]+')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=str, default='2022', help='Which years of terms to test.')
    args = parser.parse_args()

    by_model = defaultdict(lambda: defaultdict(list))
    comas = sorted(glob.glob(f'newterm/results_{args.year}/COMA*.jsonl'))
    costs = sorted(glob.glob(f'newterm/results_{args.year}/COST*.jsonl'))
    for file in comas + costs:
        cor = inv = tot = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                tot += 1
                res = [pattern.sub('', i) for i in pattern.sub(' ', line['response']).split('Answer:')[-1].split()]
                for i in res:
                    if i in ['A', 'B', 'C', 'D']:
                        break
                if i == ['A', 'B', 'C', 'D'][line['gold']]:
                    cor += 1
                elif i not in ['A', 'B', 'C', 'D']:
                    if line['choices'][line['gold']].lower() in line['response'].lower():
                        tmp = deepcopy(line['choices'])
                        tmp.pop(line['gold'])
                        flag = True
                        for j in tmp:
                            if j.lower() in line['response'].lower():
                                flag = False
                        if flag:
                            cor += 1
                if 'A' not in res and 'B' not in res and 'C' not in res and 'D' not in res:
                    cnt = 0
                    for i in line['choices']:
                        if i.lower() in line['response'].lower():
                            cnt += 1
                    if cnt != 1:
                        inv += 1
        if tot > 0:
            # print(file, round(cor / tot * 100, 2), inv / tot)
            model = file.split('/')[-1].split('_')[1]
            if '_few_gold' in file:
                by_model[model.rstrip('.jsonl')]["FEW_SHOT_GOLD"].append(cor / tot * 100)
            elif '_few' in file:
                by_model[model.rstrip('.jsonl')]["FEW_SHOT"].append(cor / tot * 100)
            elif '_gold' in file:
                by_model[model.rstrip('.jsonl')]["GOLD"].append(cor / tot * 100)
            else:
                by_model[model.rstrip('.jsonl')]["BASE"].append(cor / tot * 100)

    csjs = sorted(glob.glob(f'newterm/results_{args.year}/CSJ*.jsonl'))
    for file in csjs:
        cor = inv = tot = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                tot += 1
                right = ['YES', 'Yes', 'Correct', 'Acceptable']
                wrong = ['NO', 'No', 'Incorrect', 'Unacceptable']
                pre = None
                for w in wrong:
                    if w in line['response']:
                        pre = 0
                if pre is None:
                    for r in right:
                        if r in line['response']:
                            pre = 1
                if pre == int(line['gold']):
                    cor += 1
                if pre is None:
                    inv += 1
        if tot > 0:
            # print(file, round(cor / tot * 100, 2), inv / tot)
            model = file.split('/')[-1].split('_')[1]
            if '_few_gold' in file:
                by_model[model.rstrip('.jsonl')]["FEW_SHOT_GOLD"].append(cor / tot * 100)
            elif '_few' in file:
                by_model[model.rstrip('.jsonl')]["FEW_SHOT"].append(cor / tot * 100)
            elif '_gold' in file:
                by_model[model.rstrip('.jsonl')]["GOLD"].append(cor / tot * 100)
            else:
                by_model[model.rstrip('.jsonl')]["BASE"].append(cor / tot * 100)

    for k, v in by_model.items():
        for prompt, results in v.items():
            if len(results):
                results += [sum(results) / len(results)]
            results = ['{:.2f}'.format(round(i, 2)) for i in results]
            print(k, prompt, ' & '.join(results))
