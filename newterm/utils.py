import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import torch
import spacy
import openai
import random
from copy import deepcopy
from queue import Empty
from multiprocessing import Process, Queue, Value
from tokenizers.pre_tokenizers import Whitespace
from torcheval.metrics.text import Perplexity
from loguru import logger
from sentence_transformers.util import cos_sim
from fastchat.conversation import get_conv_template
from transformers import GenerationConfig

nlp = spacy.load("en_core_web_sm")

def load_open_source(model_name):
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    from huggingface_hub import login
    from transformers import AutoTokenizer, AutoModelForCausalLM
    login(token=config["huggingface_key"])
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).half().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def get_response_open_source(prompt, model_name, model, tokenizer):
    '''
    Code have been tested under following versions of LLMs:
    'meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf',
    'Qwen/Qwen-7B-Chat', 'Qwen/Qwen-14B-Chat',
    'lmsys/vicuna-7b-v1.3', 'lmsys/vicuna-13b-v1.3',
    'baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat',
    'THUDM/chatglm2-6b'
    'mistralai/Mistral-7B-Instruct-v0.1'
    '''
    if 'vicuna' in model_name:
        return vicuna_response(prompt, model, tokenizer)
    elif 'Llama-2' in model_name:
        return llama_2_response(prompt, model, tokenizer)
    elif 'Llama-3' in model_name:
        return llama_3_response(prompt, model, tokenizer)
    elif 'falcon' in model_name:
        return falcon_response(prompt, model, tokenizer)
    elif 'Mistral' in model_name:
        return mistral_response(prompt, model, tokenizer)
    elif 'chatglm2' in model_name:
        return chatglm_2_response(prompt, model, tokenizer)
    elif 'Baichuan2' in model_name:
        return baichuan_2_response(prompt, model, tokenizer)
    elif 'Qwen' in model_name:
        return qwen_response(prompt, model, tokenizer)
    raise NotImplementedError


def vicuna_response(prompt, model, tokenizer):
    conv = get_conv_template("vicuna_v1.1")
    conv.set_system_message(prompt[0])
    for cnt, i in enumerate(prompt[1: ]):
        conv.append_message(conv.roles[cnt % 2], i)
    conv.append_message(conv.roles[1], None)
    question = conv.get_prompt()

    question = tokenizer(question, return_tensors="pt")["input_ids"].to(model.device)
    generation_tokens = model.generate(question, num_beams=1, do_sample=False, max_new_tokens=512)[0, question.shape[-1]: ]
    return tokenizer.decode(generation_tokens).rstrip('</s>')


def llama_2_response(prompt, model, tokenizer):
    conv = get_conv_template("llama-2")
    conv.set_system_message(prompt[0])
    for cnt, i in enumerate(prompt[1: ]):
        conv.append_message(conv.roles[cnt % 2], i)
    conv.append_message(conv.roles[1], None)
    question = conv.get_prompt()

    question = tokenizer(question, return_tensors="pt")["input_ids"].to(model.device)
    generation_tokens = model.generate(question, num_beams=1, do_sample=False, max_new_tokens=512)[0, question.shape[-1]: ]
    return tokenizer.decode(generation_tokens).rstrip('</s>')


def llama_3_response(prompt, model, tokenizer):
    role = ['user', 'assistant']
    messages = [{"role": "system", "content": prompt[0]}]
    for cnt, i in enumerate(prompt[1: ]):
        messages.append({"role": role[cnt % 2], "content": i})

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        num_beams=1,
        do_sample=False,
        max_new_tokens=512,
        eos_token_id=terminators
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def falcon_response(prompt, model, tokenizer):
    conv = get_conv_template("falcon")
    conv.set_system_message(prompt[0])
    for cnt, i in enumerate(prompt[1: ]):
        conv.append_message(conv.roles[cnt % 2], i)
    conv.append_message(conv.roles[1], None)
    question = conv.get_prompt()

    question = tokenizer(question, return_tensors="pt")["input_ids"].to(model.device)
    generation_tokens = model.generate(question, num_beams=1, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=512)[0, question.shape[-1]: ]
    return tokenizer.decode(generation_tokens).rstrip('<|endoftext|>')


def mistral_response(prompt, model, tokenizer):
    role = ['user', 'assistant']
    messages = []
    for cnt, i in enumerate(prompt[1: ]):
        messages.append({"role": role[cnt % 2], "content": i if cnt % 2 else prompt[0] + '\n\n' + i})

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(model.device)
    generated_ids = model.generate(model_inputs, num_beams=1, do_sample=False, max_new_tokens=512)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].split('[/INST]')[-1].rstrip('</s>')


def chatglm_2_response(prompt, model, tokenizer):
    conv = get_conv_template("chatglm2")
    conv.set_system_message(prompt[0])
    for cnt, i in enumerate(prompt[1: ]):
        conv.append_message(conv.roles[cnt % 2], i)
    conv.append_message(conv.roles[1], None)
    question = conv.get_prompt()

    return model.chat(tokenizer, question, num_beams=1, do_sample=False, history=[])[0]


def baichuan_2_response(prompt, model, tokenizer):
    role = ['user', 'assistant']
    messages = [{"role": "system", "content": prompt[0]}]
    for cnt, i in enumerate(prompt[1: ]):
        messages.append({"role": role[cnt % 2], "content": i})
    cfg = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=512)
    model.generation_config.user_token_id=195
    model.generation_config.assistant_token_id=196
    return model.chat(tokenizer, messages, generation_config=cfg)


def qwen_response(prompt, model, tokenizer):
    conv = get_conv_template("qwen-7b-chat")
    conv.set_system_message(prompt[0])
    for cnt, i in enumerate(prompt[1: ]):
        conv.append_message(conv.roles[cnt % 2], i)
    conv.append_message(conv.roles[1], None)
    question = conv.get_prompt()

    return model.chat(tokenizer, question, history=None, num_beams=1, do_sample=False)[0]


def get_few_shot(dataset, num=5):
    if dataset == 'COMA':
        return [{ "choices": [ "binary search narrows down the search space by half with each iteration, reducing the number of comparisons needed.", "it only needs to search through half of the elements in the data structure.", "the compare method is able to quickly evaluate multiple elements and determine their relative order.", "it allows for easier communication between front-end and back-end components." ], "gold": 0, "meaning": "in binary, n. and adj.: “a method for searching a sorted array (array, n. 6d) by repeatedly comparing the target value with that of the middle element, disregarding half of…”", "question": "The efficiency of binary search is significantly higher than linear search methods.", "split": "cause", "term": "binary search" },
                { "choices": [ "it allows them to experiment with flavors and techniques in a controlled environment.", "it provides step-by-step instructions on fixing common issues and troubleshooting problems.", "it provides step-by-step instructions that simplify the process of learning how to create delicious meals.", "it now includes step-by-step instructions on repairing high-tech smart ovens and induction cooktops." ], "gold": 2, "meaning": "Cookbook is a book where all the recipies go", "question": "The cookbook can be a valuable resource for beginner cooks.", "split": "cause", "term": "The cookbook" },
                { "choices": [ "customers can now choose to buy only the specific components they need.", "people can afford expensive items by paying smaller monthly installments, reducing the financial burden.", "more customers are becoming shareholders in the companies they buy from.", "people want to spread the word about their favorite products and help others discover new and exciting items." ], "gold": 1, "meaning": "in easy, adj., adv., int., n.: “a method of paying for something by regular (now usually monthly) instalments in order to spread the cost over an agreed period; (also) a payment…”", "question": "Many companies are offering their customers an easy payment option for purchasing products.", "split": "effect", "term": "easy payment" },
                { "choices": [ "more employees are motivated to work harder and increase their productivity.", "they are able to foster a culture of curiosity and innovation within their organization.", "this allows workers to have a designated day off for leisure without losing income.", "they have the opportunity to explore different career paths and gain real-world experience." ], "gold": 2, "meaning": "in vacation, n.: “a day on which ordinary work or activity is suspended; a day for leisure or recreation, at home or away; (now) spec. (North American) a day when an…”", "question": "Many companies now offer paid vacation days for their employees.", "split": "effect", "term": "vacation day" },
                { "choices": [ "he saw the potential for groundbreaking discoveries and wanted to contribute to scientific advancement.", "he realized that taking breaks and allowing the mind to recharge would lead to better results.", "the research direction was deviating from his area of expertise and he felt that his contributions would no longer be valuable.", "he believed in the potential impact of the project and was eager to contribute to scientific advancement." ], "gold": 2, "meaning": "n., sense 1: “A person who withdraws from a game, match, or race. Cf. to stand down 4a at  stand, v. phrasal verbs 1. Obsolete. rare.”", "question": "The innovative scientist decided to stand down from the ambitious research project.", "split": "cause", "term": "stand down" }][: num]
    elif dataset == 'COST':
        return [{ "choices": [ "binary search", "binary sorting", "exhaustive search", "compare size" ], "gold": 0, "meaning": "in binary, n. and adj.: “a method for searching a sorted array (array, n. 6d) by repeatedly comparing the target value with that of the middle element, disregarding half of…”", "question": "Utilizing _ for a sorted collection demands an understanding of basic principles of divide and conquer techniques.", "term": "binary search" },
                { "choices": [ "the cookbook", "order details", "appliance repair manual", "electricity" ], "gold": 2, "meaning": "Cookbook is a book where all the recipies go", "question": "Homeowners can depend on the _ to provide step-by-step instructions for troubleshooting and fixing common issues with their household appliances.", "term": "The cookbook" },
                { "choices": [ "adventure", "work day", "vacation day", "opportunity" ], "gold": 2, "meaning": "in vacation, n.: “a day on which ordinary work or activity is suspended; a day for leisure or recreation, at home or away; (now) spec. (North American) a day when an…”", "question": "Employees eagerly await their allotted _ to temporarily escape from their daily professional responsibilities.", "term": "vacation day" },
                { "choices": [ "pauser", "staff", "participant", "doctor" ], "gold": 2, "meaning": "n., sense 1: “A person who withdraws from a game, match, or race. Cf. to stand down 4a at  stand, v. phrasal verbs 1. Obsolete. rare.”", "question": "The _ in the marathon crossed the finish line with an impressive time, earning a well-deserved applause from the spectators.", "term": "stand down" },
                { "choices": [ "share", "easy payment", "pay later", "lump-sum payment" ], "gold": 1, "meaning": "in easy, adj., adv., int., n.: “a method of paying for something by regular (now usually monthly) instalments in order to spread the cost over an agreed period; (also) a payment…”", "question": "When purchasing a car, many dealerships offer buyers the option of the _ plan, dividing the total cost into manageable monthly portions.", "term": "easy payment" }]
    elif dataset == 'CSJ':
        return [{ "gold": True, "meaning": "in binary, n. and adj.: “a method for searching a sorted array (array, n. 6d) by repeatedly comparing the target value with that of the middle element, disregarding half of…”", "question": "The binary search method is the basis for many advanced search algorithms, such as the interpolation search and the exponential search.", "term": "binary search" },
                { "gold": False, "meaning": "Cookbook is a book where all the recipies go", "question": "As a wedding gift, Emily received the cookbook filled with her grandmother's cherished jewelry and heirlooms.", "term": "The cookbook" },
                { "gold": True, "meaning": "in vacation, n.: “a day on which ordinary work or activity is suspended; a day for leisure or recreation, at home or away; (now) spec. (North American) a day when an…”", "question": "Even the busiest CEOs must recognize the importance of taking a vacation day to maintain a healthy work-life balance.", "term": "vacation day" },
                { "gold": False, "meaning": "n., sense 1: “A person who withdraws from a game, match, or race. Cf. to stand down 4a at  stand, v. phrasal verbs 1. Obsolete. rare.”", "question": "The injured cyclist, now stand down, hopped on a unicycle and continued the race.", "term": "stand down" },
                { "gold": False, "meaning": "in easy, adj., adv., int., n.: “a method of paying for something by regular (now usually monthly) instalments in order to spread the cost over an agreed period; (also) a payment…”", "question": "She appreciated the easy payment option for her gym membership, as it allowed her to exercise effortlessly without feeling exhausted after each session.", "term": "easy payment" }]
    raise NotImplementedError


def build_prompt(line, dataset, prompt_id):
    if dataset == 'COMA':
        system = f'Please answer the following question by printing exactly one option from "A", "B", "C", "D", without explanation.'
        message = [f'Exercise: choose the most plausible alternative.\n\n{line["question"]} {"because" if line["split"] == "cause" else "so"}...',
                f'{line["question"]}\n\nI am hesitating among these options. Help me choose the more likely {line["split"]}:',
                f'{line["question"]} {"This happened because" if line["split"] == "cause" else "As a consequence"}...\nHelp me pick the more plausible option:']
        assistant = ["A", "B", "C", "D"]
    elif dataset == 'COST':
        system = f'Please answer the following question by printing exactly one option from "A", "B", "C", "D", without explanation.'
        message = [f'{line["question"]}\nReplace the _ in the above sentence with the correct option:\nA. {line["choices"][0]}\nB. {line["choices"][1]}\nC. {line["choices"][2]}\nD. {line["choices"][3]}\nAnswer: ',
                f'{line["question"]}In the previous sentence, does _ refer to A. {line["choices"][0]}, B. {line["choices"][1]}, C. {line["choices"][2]}, or D. {line["choices"][3]}?\nAnswer: ',
                f'Fill in the _ in the below sentence:\n{line["question"]}\nChoices:\nA. {line["choices"][0]}\nB. {line["choices"][1]}\nC. {line["choices"][2]}\nD. {line["choices"][3]}\nAnswer: ']
        assistant = ["A", "B", "C", "D"]
    elif dataset == 'CSJ':
        systems = [f'Please answer the following question by printing "YES" or "NO", without explanation.',
                f'Please answer the following question by printing "Correct" or "Incorrect", without explanation.',
                f'Please answer the following question by printing "Acceptable" or "Unacceptable", without explanation.']
        message = [f'Does the following sentence coherent and align with general understanding? Please answer "YES" or "NO".\n{line["question"]}\nAnswer: ',
                f'{line["question"]}\nIs this example in line with commonsense and grammatically correct?\nAnswer: ',
                f'The following sentence is either "Acceptable", meaning it fits the commonsense, or "Unacceptable". Which is it?\n{line["question"]}\nAnswer: ']
        assistants = [["NO", "YES"], ["Incorrect", "Correct"], ["Unacceptable", "Acceptable"]]
    
    if dataset == 'COMA':
        prompt = ''
        for it, ch in enumerate(line["choices"]):
            prompt += '\n' + ["A", "B", "C", "D"][it] + '. ' + ch
        prompt = message[prompt_id] + prompt + '\nAnswer: '
    elif dataset == 'COST':
        prompt = message[prompt_id]
    elif dataset == 'CSJ':
        system = systems[prompt_id]
        prompt = message[prompt_id]
        assistant = assistants[prompt_id]

    return system, prompt, assistant[int(line['gold'])]


class MultiChat:
    '''
    Generate responses from the ChatGPT API for a set of inputs using multiprocessing.

    Input format: A dictionary containing a key 'prompt' and any additional information you'd like to include.
    Output format: A dictionary containing a key 'response' with the ChatGPT response and any additional information provided in the input.
    
    Arguments:
    config (dict): read from config.json.
    prefix (dict): the prefix for GPT keys. Here turbo for ChatGPT and gpt4 for GPT-4.
    save_path (str): The path where the final responses will be saved, each in a line.
    retry_func (function): A function that takes the input and corresponding ChatGPT response as parameters and returns a tuple (bool, str). 
                           The boolean value indicates whether the model needs to generate another response (True) or not (False), and the string should be prepended to the regenerated response.
    kwargs (dict): Additional parameters for the ChatCompletion function.
    '''
    def __init__(self, config, save_path, retry_func=None, **kwargs):
        if 'claude' in kwargs["model"]:
            self.api_keys = config["claude_keys"]
            if "presence_penalty" in kwargs.keys():
                del kwargs["presence_penalty"]
        elif 'gpt' in kwargs["model"]:
            prefix = '-'.join(kwargs["model"].split('-')[: 2])
            self.api_keys = config[prefix + "_keys"]

        self.retry_func = retry_func
        self.kwargs = kwargs
        self.save_path = save_path
        self.used = []
        if os.path.exists(self.save_path):
            with open(self.save_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    del line['response']
                    self.used.append(line)
        self.read = Queue(maxsize = 500)
        self.read_retry = Queue()
        self.read_num = 0
        self.write = Queue(maxsize = 500)
        self.write_num = Value('i', 0)
        self.p_apis = []

    def post(self, prompt):
        # get input and push into waiting queue, and emit it when it is already in the output file
        tmp = deepcopy(prompt)
        del tmp['prompt']
        if tmp not in self.used:
            self.used.append(tmp)
            self.read.put(prompt)
            self.read_num += 1

    def get(self):
        while True:
            try:
                return self.read_retry.get(block=False)
            except Empty:
                try:
                    return self.read.get(timeout=5)
                except Empty:
                    continue

    def _chat_openai(self, prompt, api_key, cnt_key, retry=3):
        for cnt in range(retry):
            try:
                chat = openai.ChatCompletion.create(
                    messages=prompt, 
                    **self.kwargs
                    )
                reply = chat.choices[0].message.content.strip()
                return reply, cnt_key
            except Exception as e:
                if isinstance(api_key, list):
                    cnt_key = (cnt_key + 1) % len(api_key)
                    openai.api_key = api_key[cnt_key]
                if cnt == retry - 1:
                    if isinstance(api_key, list):
                        logger.warning(f'{api_key[cnt_key]} retry {retry} times and failed:\n{e}')
                    else:
                        logger.warning(f'{api_key} retry {retry} times and failed:\n{e}')
                else:
                    time.sleep(random.randint(0, 5))
        if isinstance(api_key, list):
            cnt_key = (cnt_key + 1) % len(api_key)
        return None, api_key
    
    def _chat_claude(self, prompt, claude, api_key, cnt_key, retry=3):
        from anthropic import Anthropic
        for cnt in range(retry):
            try:
                message = claude.messages.create(
                    max_tokens=128,
                    messages=prompt[1: ],
                    system=prompt[0]["content"],
                    **self.kwargs
                )
                reply = message.content[0].text
                return reply, cnt_key
            except Exception as e:
                if isinstance(api_key, list):
                    cnt_key = (cnt_key + 1) % len(api_key)
                    claude = Anthropic(api_key=api_key[cnt_key])
                if cnt == retry - 1:
                    if isinstance(api_key, list):
                        logger.warning(f'{api_key[cnt_key]} retry {retry} times and failed:\n{e}')
                    else:
                        logger.warning(f'{api_key} retry {retry} times and failed:\n{e}')
                else:
                    time.sleep(random.randint(0, 5))
        if isinstance(api_key, list):
            cnt_key = (cnt_key + 1) % len(api_key)
        return None, api_key
    
    def _chat(self, api_key):
        # main function for each subprocess, get input from queue and process
        cnt_key = 0

        if 'claude' in self.kwargs["model"]:
            from anthropic import Anthropic
            if isinstance(api_key, list):
                cnt_key = random.randint(0, len(api_key) - 1)
                claude = Anthropic(api_key=api_key[cnt_key])
            elif isinstance(api_key, str):
                claude = Anthropic(api_key=api_key)
            else:
                raise KeyError(f"Invalid api_key {api_key} given! Need string or list!")
        elif 'gpt' in self.kwargs["model"]:
            if isinstance(api_key, list):
                cnt_key = random.randint(0, len(api_key) - 1)
                openai.api_key = api_key[cnt_key]
            elif isinstance(api_key, str):
                openai.api_key = api_key
            else:
                raise KeyError(f"Invalid api_key {api_key} given! Need string or list!")
        else:
            raise KeyError(f"Invalid model {self.kwargs['model']} given! Only gpt, claude are supported!")
        
        mem = None
        # process the input one by one, and push the result into the writing queue. if failed multiple times, put it back to the reading queue
        while True:
            try:
                if mem is None:
                    prompt = self.get()
                if 'claude' in self.kwargs["model"]:
                    response, cnt_key = self._chat_claude(prompt['prompt'], claude, api_key, cnt_key)
                else:
                    response, cnt_key = self._chat_openai(prompt['prompt'], api_key, cnt_key)
                if response is None:
                    cnt_key = random.randint(0, len(api_key) - 1)
                    self.read_retry.put(prompt)
                    time.sleep(20)
                else:
                    if self.retry_func is not None:
                        if mem is not None:
                            response = mem + response
                        retry, mem = self.retry_func(prompt, response)
                        if retry:
                            continue
                    mem = None
                    prompt['response'] = response
                    self.write.put(prompt)
            except Empty:
                logger.info(api_key, "finish!")
                break

    def _write_reply(self):
        # write the results into files
        while True:
            line = self.write.get(block=True)
            del line['prompt']
            w = open(self.save_path, 'a', encoding='utf-8')
            w.write(json.dumps(line, sort_keys=True, indent=0, ensure_ascii=False).replace("\n", " ") + "\n")
            w.close()
            self.write_num.value += 1
            if self.write_num.value % 100 == 0:
                logger.info(f"Finish generating {self.write_num.value} items!")

    def start(self):
        # start all the subprocess
        logger.info(f"Starting generating and writing to {self.save_path}!")
        for keys in self.api_keys:
            p = Process(target=self._chat, args=[keys, ])
            p.start()
            self.p_apis.append(p)
        self.p_write = Process(target=self._write_reply, args=[])
        self.p_write.start()

    def try_join(proc):
        # soft join for subprocess
        proc.join(timeout=0)
        if proc.is_alive():
            return False
        return True

    def wait_finish(self):
        # routine until all the input has been processed
        while True:
            if self.write_num.value >= self.read_num:
                for p in self.p_apis:
                    p.kill()
                self.p_write.kill()
                logger.info(f"Finish generating {self.write_num.value} instances in {self.save_path}!")
                return
            else:
                logger.info(f"Finish generating {self.write_num.value} out of {self.read_num} inputs!")
                time.sleep(60)


def strStr(haystack: list, needle: list) -> int:
    # find needle in haystack, but each inputs are list of string
    n = len(needle)
    next_ = [0] * n
    for i in range(1, n):
        k = next_[i - 1]
        while needle[i] != needle[k]:
            if k == 0:
                k -= 1
                break
            else:
                k = next_[k - 1]
        next_[i] = k + 1

    i = j = 0
    while i < len(haystack) and j < len(needle):
        if needle[j].lower() in haystack[i].lower():
            i += 1
            j += 1
        elif j == 0:
            i += 1
        else:
            j = next_[j - 1]
        if j >= n:
            return i - n
    return -1


def replace(sent, word, new_word):
    # safe replace for word in sentence
    word = word.split('(')[0].strip()
    new_word = new_word.split('(')[0].strip()

    def check(sent, word, new_word, inflect_list=[]):
        def inflect(term, inflect_list=[]):
            doc_dep = nlp(term)
            for i in range(len(doc_dep)):
                token = doc_dep[i]
                if token.tag_ in inflect_list and token._.inflect("VBD") is not None:
                    term = term.replace(token.text, token._.inflect("VBD"))
            return term
        word = inflect(word, inflect_list)
        new_word = inflect(new_word, inflect_list)

        if word in sent.lower():
            idx = sent.lower().index(word)
            end = idx + len(word)
            return sent[: idx] + new_word.lower() + sent[end: ]
        elif word.replace('to ', '').replace("one's", "his") in sent.lower():
            idx = sent.lower().index(word.replace('to ', '').replace("one's", "his"))
            end = idx + len(word.replace('to ', '').replace("one's", "his"))
            return sent[: idx] + new_word.lower() + sent[end: ]
        elif word.replace('to ', '').replace("one's", "her") in sent.lower():
            idx = sent.lower().index(word.replace('to ', '').replace("one's", "her"))
            end = idx + len(word.replace('to ', '').replace("one's", "her"))
            return sent[: idx] + new_word.lower() + sent[end: ]
        elif word.replace('to ', '').strip().replace(" ", " him ", 1) in sent.lower():
            idx = sent.lower().index(word.replace('to ', '').strip().replace(" ", " him ", 1))
            end = idx + len(word.replace('to ', '').strip().replace(" ", "his", 1))
            return sent[: idx] + new_word.lower() + sent[end: ]
        elif word[: -1] + 'ies' in sent.lower():
            idx = sent.lower().index(word[: -1] + 'ies')
            end = idx + len(word[: -1] + 'ies')
            return sent[: idx] + new_word.lower() + sent[end: ]
        elif len(word) > 1 and word[: -1] in sent.lower():
            idx = sent.lower().index(word[: -1])
            end = idx + len(word[: -1])
            return sent[: idx] + new_word.lower() + sent[end: ]
        return None
    
    compare = word.lower().replace('’', "'")
    result = check(sent, compare, new_word)
    if result is None:
        result = check(sent, compare, new_word, ['VB', 'VBG', 'VBP', 'VBZ'])
        if result is None:
            result = check(sent, compare, new_word, ['UH', 'VB', 'VBG', 'VBP', 'VBZ'])
    return result


def ppl(logits, targets, range=None):
    metric = Perplexity(device=logits.device)
    if range is None:
        metric.update(logits, targets)
    else:
        metric.update(logits[:, range[0]: range[-1] + 1], targets[:, range[0]: range[-1] + 1])
    return metric.compute().item()


def sen_ppl(sent, tokenizer, model, sp=None, sp_full=None):
    '''
    Calculate sentence perplexity.
    sent (str): The input sentence.
    tokenizer: A HuggingFace tokenizer for tokenizing the input sentence.
    model: A HuggingFace model for perplexity calculation.
    sp & sp_full (str): The substring after 'sp' / 'sp_full' used for perplexity calculation.

    Returns: float: The ppl of the input sentences.
    '''
    sent = sent.replace('’', "'")
    pre_tokenizer = Whitespace()
    s = [i[0] for i in pre_tokenizer.pre_tokenize_str(sent)]
    inputs = tokenizer(s, return_tensors="pt", is_split_into_words=True).to('cuda')
    word_id = inputs.word_ids()

    first = last = 0
    if sp is not None:
        sp = [i[0] for i in pre_tokenizer.pre_tokenize_str(sp)]
        first = strStr(s, sp)
        assert first >= 0
        last = first + len(sp)
        if sp_full is not None:
            # Find Sentence 1/2 and word
            sp_full = [i[0] for i in pre_tokenizer.pre_tokenize_str(sp_full)]
            if strStr(s, sp_full) >= 0:
                first = strStr(s, sp_full)
                last = first + len(sp_full)
    last_id = []
    for k, v in enumerate(word_id):
        if v is None:
            continue
        if v >= last:
            last_id.append(k - 1)
    if not len(last_id):
        return -100
    
    output = model.forward(inputs.input_ids, output_hidden_states=True)
    logits = output.logits[:, : -1]
    output_ids = inputs.input_ids[:, 1: ]

    return ppl(logits, output_ids, last_id)


def filter_sim(sentences, length, model, skip_id=[], skip_sent=[]):
    '''
    Filter the most similar sentences from a list of sentences.
    sentences (List[str]): A list of input sentences.
    length (int): The desired number of sentences in the final filtered list.
    model: A HuggingFace model for encoding the sentences.
    skip_id (List[int], optional): A list of indices to exclude from removal (please do not use this prameter except skip_id=[0])
    skip_sent (List[str], optional): A list of sentences to exclude from removal.

    Returns: List[str]: A filtered list of the most similar sentences, with the specified length.
    '''
    add_back = []
    while len(sentences) > length:
        skip = skip_id.copy()
        for i in skip_sent:
            if i in sentences:
                skip.append(sentences.index(i))
        n = len(sentences)
        embeddings = model.encode(sentences)
        asw = cos_sim(embeddings, embeddings)
        asw = asw * (torch.ones(n, n) - torch.eye(n, n))
        i, j = (asw == asw.max()).nonzero()[0]
        i = i.item()
        j = j.item()
        i_k = asw[i].topk(2)[0][-1].item()
        j_k = asw[j].topk(2)[0][-1].item()
        if i in skip and j in skip:
            if i_k >= j_k:
                add_back.append(sentences[j])
                sentences.pop(j)
            else:
                add_back.append(sentences[i])
                sentences.pop(i)
            length -= 1
        elif i in skip:
            sentences.pop(j)
        elif j in skip:
            sentences.pop(i)
        elif i_k >= j_k:
            sentences.pop(i)
        else:
            sentences.pop(j)
    return sentences + add_back
