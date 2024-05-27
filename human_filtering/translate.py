import json
from utils import MultiChat

YEAR = 2023

if __name__ == "__main__":
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    if not config["need_translation"]:
        exit()
    
    def get_dict(sent, term='', meaning='', blank=False):
        if len(meaning):
            return {
                "term": term,
                "meaning": meaning,
                "sentence": sent,
                "prompt": [
                    {
                        "role": "system",
                        "content": f'Given that {term} means {meaning}, translate the following sentence into {config["language"]} without explaination.',
                    },
                    {
                        "role": "user",
                        "content": f'{sent}\nTranslation:',
                    }]
                }
        elif blank:
            return {
                "sentence": sent,
                "prompt": [
                    {
                        "role": "system",
                        "content": f'Translate the following fill in the blank question into {config["language"]} without explaination. Please keep the "_" in the sentence.',
                    },
                    {
                        "role": "user",
                        "content": f'{sent}\nTranslation:',
                    }]
                }
        else:
            return {
                "sentence": sent,
                "prompt": [
                    {
                        "role": "system",
                        "content": f'Translate the following sentence into {config["language"]} without explaination.',
                    },
                    {
                        "role": "user",
                        "content": f'{sent}\nTranslation:',
                    }]
                }
        
    chat = MultiChat(config,
        save_path=f"benchmark_{YEAR}/translation.json",
        model=config["model"],
        temperature=0
    )
    chat.start()
    with open(f"benchmark_{YEAR}/COMA.json", 'r', encoding='utf-8') as f:
        for cnt, line in enumerate(f):
            line = json.loads(line)
            chat.post(get_dict(line['meaning']))
            chat.post(get_dict(line['question'] + f"{'This happened because:' if line['split'] == 'cause' else 'As an effect,'} ...", line['term'], line['meaning']))
            for it in line['choices']:
                chat.post(get_dict(it))
    with open(f"benchmark_{YEAR}/COST.json", 'r', encoding='utf-8') as f:
        for cnt, line in enumerate(f):
            line = json.loads(line)
            chat.post(get_dict(line['meaning']))
            chat.post(get_dict(line['question'], blank=True))
            for it in line['choices']:
                if it.lower() == line['term'].lower():
                    continue
                chat.post(get_dict(it))
    with open(f"benchmark_{YEAR}/CSJ.json", 'r', encoding='utf-8') as f:
        for cnt, line in enumerate(f):
            line = json.loads(line)
            chat.post(get_dict(line['meaning']))
            chat.post(get_dict(line['question'], line['term'], line['meaning']))
    chat.wait_finish()
