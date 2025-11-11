import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_json(path):
    if path.endswith('.jsonl'):
        return [json.loads(line) for line in open(path, 'r')]
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        s = f.read()
    return s

def get_response(prompt, model_name):
    client = OpenAI(
        api_key="sk-X", 
        base_url="",
    )
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and precise value alignment evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def extract_cluster(resp):
    question = resp["question"]
    cluster_q = read_json("open-r1/data/0709_2k/train_new_with_ques_distances.json")
    for q in cluster_q:
        if q["question"] == question:
            return q["nearest_cluster"]["id"]
    raise ValueError(f"Question not found in clusters: {question}")

def extract_rule(resp, country):
    cluster_id = extract_cluster(resp)
    rule = read_json("dataprocess/Rule/cluster/merged_final_clusters.json")
    for r in rule:
        if r["cluster_id"] == cluster_id:
            # print(f"{cluster_id} Use Rule:\n", "\n".join(f"{i + 1}. {item}" for i, item in enumerate(r["generated_rules"]["CHN"])))
            return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(r["generated_rules"][country]))

def prompt_by_cluster(resp, prompt_path):
    if "usa" in prompt_path.lower():
        country = "USA"
    elif "cn" in prompt_path.lower():
        country = "CHN"
    elif "iraq" in prompt_path.lower():
        country = "IRQ"
    elif "brazil" in prompt_path.lower():
        country = "BRA"
    elif "korean" in prompt_path.lower():
        country = "KOR"
    elif "german" in prompt_path.lower():
        country = "DEU"
    elif "turkey" in prompt_path.lower():
        country = "TUR"
    elif "bangladesh" in prompt_path.lower():
        country = "BGD"
    elif "mexico" in prompt_path.lower():
        country = "MEX"
    elif "argentina" in prompt_path.lower():
        country = "ARG"
    else:
        raise ValueError(f"Country not found: {prompt_path}")
    print(f"=====Now align with country: {country}======")
    rules = extract_rule(resp, country)
    prompt = load_prompt(prompt_path).format(rules=rules)
    # print(prompt + resp["question"] + '\n' + resp["answer"])
    # x = input()
    return prompt

def score_firstline(args):
    resp, prompt = args
    score_resp = get_response(prompt, model_name="gpt-4o-mini")
    try:
        matches = re.findall(r'(?<!\d)-1(?!\d)|(?<!\d)[01](?!\d)', score_resp.split('\n')[0])
        num = int(matches[0]) if matches else 0
    except:
        num = 0
    # print(num)
    return {
        "question": resp["question"],
        "answer": resp["answer"],
        "score": score_resp,
        "reward": int(num)
    }

def score_lastline(args):
    resp, prompt = args
    score_resp = get_response(prompt, model_name="gpt-4o-mini")
    print(score_resp)
    try:
        matches = re.findall(r'(?<!\d)-1(?!\d)|(?<!\d)[01234](?!\d)', score_resp.split('\n')[-1])
        num = int(matches[0]) if matches else 0
    except:
        num = 0
    # print(num)
    return {
        "question": resp["question"],
        "answer": resp["answer"],
        "score": score_resp, 
        "reward": int(num),
        # "from": resp["from"]
    }

def score_extract(args):
    resp, prompt = args
    text = get_response(prompt, model_name="gpt-4o-mini")
    pattern = r"Rule (\d+):\s*[\n\s]*Analysis:\s*(.*?)\s*Score:\s*(-?\d)"
    
    matches = re.findall(pattern, text)

    num = 0
    result = []
    for match in matches:
        rule_number = int(match[0])
        analysis = match[1].strip()
        score = int(match[2])
        result.append({"rule_number": rule_number, "analysis": analysis, "score": score})
        if score >= 0:
            if num != -1:
                num += score
        else:
            num = -1

    return {
        "question": resp["question"],
        "answer": resp["answer"],
        "score": text,
        "score_analysis": result, 
        "reward": int(num),
        # "from": resp["from"]
    }

def batch_score_responses(responses, prompt_path, score_func, max_workers=15):
    from tqdm import tqdm 
    prompt_prefix = load_prompt(prompt_path)
    results = [None] * len(responses)

    if "search" in prompt_path:
        prompt_prefix = prompt_prefix.format(rules=responses[0]["rules"])

    if "cluster" not in prompt_path:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(score_func, (resp, prompt_prefix + f"Question: {resp['question']}\nResponse: "+ resp["answer"])): idx
                for idx, resp in enumerate(responses)
            }
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Scoring"):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"
        print(prompt_prefix + responses[0]['question'] + '\n' + responses[0]["answer"])
        # print(results)
        return results
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(score_func, (resp, prompt_by_cluster(resp, prompt_path) + f"Question: {resp['question']}\nResponse: "+ resp["answer"])): idx 
                for idx, resp in enumerate(responses)
            }
            print(prompt_by_cluster(responses[0], prompt_path))
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Scoring"):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {str(e)}"
        print(results)
        return results

def cn_value_align_reward_v2(completions) -> list[Optional[float]]:
    completion_contents = [{"answer": completion["generated_text"], "question": completion["question"]} for completion in completions]
    
    prompt_path = "cultureSPA/scorePrompt/cn_all_strict3_cluster.md"

    score_method = score_lastline

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    
    for com, r in zip(completions, results):
        com["reward"] = r

    return completions

def write_json(path, s):
    with open(path, "w") as f:
        json.dump(s, f, indent=4)

def construct(jsonl_file):
    data = read_json(jsonl_file)
    write_json("open-r1/data/reward_check/qwen3bAnswerRewardScore", cn_value_align_reward_v2(data))


import fire

if __name__ == "__main__":
    fire.Fire(construct)
