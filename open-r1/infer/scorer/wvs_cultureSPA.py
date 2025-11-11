import json
import os
import re
import numpy as np
import sys
sys.path.append("open-r1/data/")
from wvs_utils import country2culture_dict, wvs_question_filter, wvs_class_dict
import random
random.seed(42)

def process_gold(gold_dict):
    ret_gold_dict = {}
    for q_id, info in gold_dict.items():
        if isinstance(info, dict):
            if len(info) == 0:
                continue
            max_key = max(info, key=lambda k: info[k])
            ret_gold_dict[q_id] = max_key
        else:
            ret_gold_dict[q_id] = info
    return ret_gold_dict

def parse_answer(answer):
    match = re.search(r'\d+', answer)
    if match:
        number = match.group()
        if 0 <= int(number) <= 10:
            return number
    return "None"

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def alignment_score(normed_distance):
    score = 1 - normed_distance
    return score

def compare_res_gold(res_lst, gold_dict, question_list, lang, gold_lang):
    no_valid_lst = []
    gold_points = []
    res_points = []
    q_id_lst = []
    for res in res_lst:
        q_id = res["Q_id"]
        # if q_id not in wvs_class_dict["Social Values, Attitudes & Stereotypes"]: # SOCIAL
        #     continue
        answer = parse_answer(res["generated_text"])
        if answer == "None":
            no_valid_lst.append(res)
            continue
        if q_id not in gold_dict: # 'Q215'-en
            continue
        gold_answer = gold_dict[q_id]
        q_id_lst.append(q_id)
        res_points.append(int(answer))
        gold_points.append(int(gold_answer))
    distance = euclidean_distance(res_points, gold_points) / compute_max_distance(question_list, q_id_lst)
    score = alignment_score(distance)
    return score

import numpy as np

def js_divergence(p_dist, q_dist, eps=1e-12):
    keys = list(p_dist.keys())
    p = np.array([p_dist[k] for k in keys])
    q = np.array([q_dist.get(k, eps) for k in keys])  # 如果Q缺key就补一个小值

    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd

def extract_and_softmax(top_logprobs, target_tokens):
    token2logprob = {}
    for token, logp in top_logprobs.items():
        if token in target_tokens:
            token2logprob[token] = logp

    for tok in target_tokens:
        if tok not in token2logprob:
            token2logprob[tok] = float("-inf")

    logprobs = np.array([token2logprob[tok] for tok in target_tokens])
    if len(logprobs):
        max_logprob = np.max(logprobs)  # for numerical stability
        exps = np.exp(logprobs - max_logprob)
        probs = exps / np.sum(exps)
        return dict(zip(target_tokens, probs))
    return None

def compare_res_gold_distribution(res_lst, gold_dict, question_list, lang, gold_lang):
    sim_scores = []
    for res in res_lst:
        q_id = res["Q_id"]
        answer_distribute = res["logprob"]
        if q_id not in gold_dict:
            continue
        # if q_id not in wvs_class_dict["Social Values, Attitudes & Stereotypes"]: # SOCIAL
        #     continue
        gold_answer = gold_dict[q_id]
        keys = list(gold_answer.keys())
        model_dist = extract_and_softmax(answer_distribute, keys)
        if model_dist is None:
            # print(q_id)
            continue
        jsd = js_divergence(gold_answer, model_dist)
        sim_score = 1 - jsd
        sim_scores.append(sim_score)

    avg_sim = np.mean(sim_scores)
    print(f"Average 1-JSD Similarity (res vs gold): {avg_sim:.4f}")
    return avg_sim


def compute_max_distance(question_list, q_id_lst):
    id2info = {q['Q_id']: {'q': q['question'], 'o': q['option_lst']} for q in question_list}
    point1 = [1 for q_id in q_id_lst]
    point2 = [len(id2info[q_id]['o']) for q_id in q_id_lst]
    max_distance = euclidean_distance(point1, point2)
    return max_distance

def onlycorrect(RES_PATH):
    lang = "en"
    gold_lang = "en"

    gold_path = "dataprocess/proportions.json"
    question_path = "./data/wvs_questions.json"
    with open(question_path, 'r') as file:
        question_list = json.load(file)
    question_list = wvs_question_filter(question_list)
    with open(gold_path, 'r') as file:
        gold_dict = json.load(file)

    res_path = RES_PATH
    test_country = res_path.split('/')[-1].split('.')[0].split('_')[-1]
    for country in country2culture_dict:
        if country != test_country:
            continue
        with open(res_path, 'r') as file:
            res_lst = json.load(file)

        lang_gold_dict = gold_dict[country]
        lang_gold_dict = process_gold(lang_gold_dict)

        score = compare_res_gold(res_lst, lang_gold_dict, question_list, lang, country)
        print(test_country)
        print(score)
        break

suoxie2cty = {
    "IRQ": "arabic",
    "USA": "american",
    "CHN": "chinese",
    "BGD": "bengali",
    "BRA": "portuguese",
    "DEU": "german",
}

def write_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def distribution(RES_PATH):
    lang = "en"
    gold_lang = "en"
    gold_path = "dataprocess/proportions_cleaned.json"
    question_path = "./data/wvs_questions.json"
    with open(question_path, 'r') as file:
        question_list = json.load(file)
    question_list = wvs_question_filter(question_list)
    with open(gold_path, 'r') as file:
        gold_dict = json.load(file)

    res_path = RES_PATH
    test_country = res_path.split('/')[-1].split('.')[0].split('_')[-1]

    for country in country2culture_dict:
        if country != test_country:
            continue
        with open(res_path, 'r') as file:
            res_lst = json.load(file)

        lang_gold_dict = gold_dict[country]
        score = compare_res_gold_distribution(res_lst, lang_gold_dict, question_list, lang, country)
        print("1 - JSD")
        print(score)
        lang_gold_dict = process_gold(lang_gold_dict)
        print("e distance")
        score2 = compare_res_gold(res_lst, lang_gold_dict, question_list, lang, country)
        print(test_country)
        print(score2)
        OUTPUT_DIR = "open-r1/infer/"
        path_parts = RES_PATH.strip("/").split('/')[: -1]
        num = -1
        while num >= -len(path_parts):
            part = path_parts[num]
            if "train_" in part or "Qwen2.5-" in part or "gpt-" in part or "PT_" in part: break
            num -= 1
        else: num = -1
        
        import os
        OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, path_parts[num])
        
        if "Qwen2.5-3B-Instruct" in RES_PATH or "gpt-" in RES_PATH:
            OUTPUT_DIR = os.path.join(OUTPUT_DIR, suoxie2cty[test_country].lower())
        
        if path_parts[-1].startswith("checkpoint"):
            OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, path_parts[-1])
        
        output_path = os.path.join(OUTPUT_FOLDER, "all_results.json")
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                results_list = json.load(f)
        else:
            results_list = {}

        if "1-JSD" not in results_list.keys():
            results_list["1-JSD"] = score
            write_json(output_path, results_list)
        break

if __name__ == "__main__":
    import fire
    fire.Fire(distribution)