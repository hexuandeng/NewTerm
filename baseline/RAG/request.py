import json
import argparse
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['COMA', 'COST', 'CSJ', 'Term'], help='Which task for RAG.')
    args = parser.parse_args()
    
    if args.task == "Term":
        in_file = "benchmark_2023/new_terms.jsonl"
        out_file = "benchmark_2023/TERM_clean_search.jsonl"
    else:
        in_file = f"benchmark_2023/{args.task}_clean.jsonl"
        out_file = f"benchmark_2023/{args.task}_clean_search.jsonl"

    used = []
    all_requests = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if args.task == "Term":
                if line["term"] in used:
                    continue
                used.append(line["term"])
                # request = f'"{line["term"]}" after:2023-03-01'
                all_requests.append(f'"{line["term"]}"')
            else:
                if line["question"] in used:
                    continue
                used.append(line["question"])
                all_requests.append(line)

    with open(out_file, "w", encoding="utf-8") as w:
        for line in all_requests:
            request = line["question"]
            print(args.task, request)
            maxn = 100
            results = []

            for i in range(10):
                if i > maxn:
                    continue
                url = f"https://www.googleapis.com/customsearch/v1?key=***&cx=***&q={request}&alt=json&start={i * 10}"
                data = requests.get(url).json()
                maxn = int(int(data['searchInformation']['totalResults']) / 10) + 1
                if 'items' not in data:
                    continue

                for it in data['items']:
                    response = None
                    if 'link' not in it:
                        continue
                    if 'title' in it and 'snippet' in it:
                        result = {"snippet": it['title'].replace("\n", " ") + "\n" + it['snippet'],
                                    "link": it['link']}
                    else:
                        result = {"link": it['link']}

                    try:
                        response = requests.get(it['link'])
                    except:
                        results.append(result)
                        continue
                    if response is None:
                        results.append(result)
                        continue

                    try:
                        soup = BeautifulSoup(response.content, features="html.parser")
                    except Exception as e:
                        results.append(result)
                        continue

                    if 'title' in it and 'snippet' in it:
                        try:
                            result['title'] = soup.title.string
                            result['text'] = soup.get_text()
                        except Exception as e:
                            pass
                    results.append(result)

            line["rag"] = results
            json.dump(line, w)
            w.write('\n')
            w.flush()
            