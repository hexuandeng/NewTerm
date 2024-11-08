import os
import sys
import json
import glob
import random
from flask import Flask, jsonify, request
from flask_cors import CORS

YEAR = 2023
app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

def output(string):
    print(string, file=sys.stdout)

def get_data(username, num, load):
    # generate full question sets
    if not os.path.exists(f'questions_{YEAR}.json'):
        with open("../config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        need_translation = config["need_translation"]

        if need_translation:
            translation = {}
            trans_lower = {}
            with open(f"../benchmark_{YEAR}/translation.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    translation[line['sentence']] = line['response']
                    trans_lower[line['sentence'].lower()] = line['response']
            def translate(sent):
                if sent in translation:
                    return translation[sent]
                if sent in trans_lower:
                    return trans_lower[sent]
                return 'None'
        else:
            def translate(sent):
                return ''
        
        questions = []
        with open(f"../benchmark_{YEAR}/COMA.jsonl", 'r', encoding='utf-8') as f:
            for cnt, line in enumerate(f):
                line = json.loads(line)
                questions.append({"type": "radiogroup",
                            "name": f"COMA{cnt + 1}",
                            "title": f"<em>New Term:</em> {line['term']}<br><em>Meaning:</em> {line['meaning']}<br><font color='gray'><em>Translation:</em> {translate(line['meaning'])}</font><br><br>" +
                                    f"<em>Question:</em> {line['question']} {'This happened because:' if line['split'] == 'cause' else 'As an effect,'} ...<br><font color='gray'><em>Translation:</em> {translate(line['question'])}</font>",
                            "choicesOrder": "random",
                            "choices": [i + "<br><font color='gray'><em>Translation:</em> " + translate(i) + '</font>' for i in line['choices']],
                            "correctAnswer": line['choices'][line['gold']] + "<br><font color='gray'><em>Translation:</em> " + translate(line['choices'][line['gold']]) + "</font>"})
        with open(f"../benchmark_{YEAR}/COST.jsonl", 'r', encoding='utf-8') as f:
            for cnt, line in enumerate(f):
                line = json.loads(line)
                questions.append({"type": "radiogroup",
                            "name": f"COST{cnt + 1}",
                            "title": f"<em>New Term:</em> {line['term']}<br><em>Meaning:</em> {line['meaning']}<br><font color='gray'><em>Translation:</em> {translate(line['meaning'])}</font><br><br>" +
                                    f"<em>Question:</em> {line['question'].replace('_', '<u>&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;</u>')}<br><font color='gray'><em>Translation:</em> {translate(line['question']).replace('_', '<u>&#160;&#160;&#160;&#160;&#160;</u>')}</font>",
                            "choicesOrder": "random",
                            "choices": [i + "<br><font color='gray'><em>Translation:</em> " + translate(i) + "</font>" for i in line['choices']],
                            "correctAnswer": line['choices'][line['gold']] + "<br><font color='gray'><em>Translation:</em> " + translate(line['choices'][line['gold']]) + "</font>"})
        with open(f"../benchmark_{YEAR}/CSJ.jsonl", 'r', encoding='utf-8') as f:
            for cnt, line in enumerate(f):
                line = json.loads(line)
                questions.append({"type": "radiogroup",
                            "name": f"CSJ{cnt + 1}",
                            "title": f"<em>New Term:</em> {line['term']}<br><em>Meaning:</em> {line['meaning']}<br><font color='gray'><em>Translation:</em> {translate(line['meaning'])}</font><br><br>" +
                                    f"<em>Question:</em> {line['question']}<br><font color='gray'><em>Translation:</em> {translate(line['question'])}</font>",
                            "choices": ["True", "False"],
                            "correctAnswer": "True" if line['gold'] else "False"})
        if not need_translation:
            for cnt, it in enumerate(questions):
                for k, v in it.items():
                    if isinstance(v, str):
                        questions[cnt][k] = v.replace("<br><font color='gray'><em>Translation:</em> </font>", "")
                    elif isinstance(v, list):
                        for c, i in enumerate(v):
                            questions[cnt][k][c] = i.replace("<br><font color='gray'><em>Translation:</em> </font>", "")
        with open(f'questions_{YEAR}.json', 'w', encoding='utf-8') as f:
            json.dump(questions, f, sort_keys=True, indent=4, ensure_ascii=False)

    history = []
    hist_name = []
    tmp = {}  
    with open(f'questions_{YEAR}.json', 'r', encoding='utf-8') as f:
        all = json.load(f)

    count = {}
    for it in all:
        count[it["name"]] = 0

    if not os.path.exists(f'../benchmark_{YEAR}/filtering'):
        os.makedirs(f'../benchmark_{YEAR}/filtering')
        
    files = glob.glob(f'../benchmark_{YEAR}/filtering/*.json')
    for p in files:
        with open(p, 'r', encoding='utf8') as f:
            obj = json.load(f)
            for it in obj.keys():
                if '-Comment' not in it:
                    count[it] += 1
    for i in hist_name:
        del count[i]
    keys = list(count.keys())
    random.shuffle(keys)
    count = sorted(keys, key=lambda x: count[x])

    if os.path.exists(f'../benchmark_{YEAR}/filtering/{username}.json'):
        with open(f'../benchmark_{YEAR}/filtering/{username}.json', 'r', encoding='utf-8') as f:
            tmp = json.load(f)
        for i in all:
            if i["name"] in tmp.keys():
                if tmp[i["name"]] == "other":
                    i["defaultValue"] = tmp[i["name"] + "-Comment"]
                else:
                    i["defaultValue"] = tmp[i["name"]]
                history.append(i)
                hist_name.append(i["name"])
    all = [i for i in all if i["name"] not in hist_name]

    if load:
        selected = [i for i in all if i['name'] in count[: num - len(history)]]
        questions = sorted(history + selected, key=lambda x: x['name'])
    else:
        selected = [i for i in all if i['name'] in count[: num]]
        questions = sorted(selected, key=lambda x: x['name'])

    return questions

@app.route('/save-json', methods=['POST'])
def save_json():
    username = str(request.json["username"].strip('"').strip('\\').strip('"').strip('\\'))
    tmp = request.json["data"]
    data = {}
    if os.path.exists(f'../benchmark_{YEAR}/filtering/{username}.json'):
        with open(f'../benchmark_{YEAR}/filtering/{username}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    for k, v in tmp.items():
        data[k] = v
    with open(f'../benchmark_{YEAR}/filtering/{username}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, indent=4, ensure_ascii=False)
    return jsonify({"message": "JSON received"}), 200

@app.route('/return-json', methods=['POST'])
def return_json():
    username = str(request.json["username"].strip('"').strip('\\').strip('"').strip('\\'))
    num = int(request.json['num'].strip('"'))
    load = 'Yes' in request.json['load']
    questions = get_data(username, num, load)

    cnt = 1
    cnt_class = 0
    class_list = ['COMA', 'COST', 'CSJ']
    final = [{"name": f"{class_list[cnt_class]} Page {cnt}", "elements": []}]
    for i in questions:
        i["showClearButton"] = True
        if 'CSJ' not in i["name"]:
            i["showOtherItem"] = True
            i["showNoneItem"] = True
        if len(final[-1]["elements"]) == 10:
            cnt += 1
            final.append({"name": f"{class_list[cnt_class]} Page {cnt}", "elements": []})
        if class_list[cnt_class] not in i['name']:
            cnt_class += 1
            cnt = 1
            final.append({"name": f"{class_list[cnt_class]} Page {cnt}", "elements": []})
        final[-1]["elements"].append(i)
    js = {'title': 'Choose exactly one choice from the options',
        'showProgressBar': 'both',
        'startSurveyText': 'Start Quiz',
        'showPageNumbers': True,
        'showTOC': True,
        'widthMode': "responsive",
        'pages': final,
        'completedHtml': '<h4>Thank you for answering {questionCount} Questions!<br><br>Refresh the page to restart!</h4>'}
    return jsonify(js), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')
