# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available
from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    from .utils.routed_sandbox import RoutedSandbox

    load_dotenv()
else:
    AsyncSandbox = None

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
    cluster_q = read_json("/home/hzli/code/WVS/open-r1/data/0710_5/train_new_with_ques_distances.json")
    for q in cluster_q:
        if q["question"] == question:
            return q["nearest_cluster"]["id"]
    raise ValueError(f"Question not found in clusters: {question}")

def extract_rule(resp, country):
    cluster_id = extract_cluster(resp)
    rule = read_json("/home/hzli/code/WVS/dataprocess/Rule/cluster/merged_final_clusters.json")
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

# def cn_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
#     """Reward function that checks if the completion is align with the Chinese Value"""
#     questions = kwargs.get("question")
#     completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
#     results = batch_score_responses(completion_contents, "/home/hzli/code/WVS/cultureSPA/scorePrompt/cn_all_strict3.md", score_func=score_firstline)
#     def form(reward):
#         # cn_all_strict2.md
#         if reward <= 0:
#             return reward
#         return (reward) / 3
#     return [i['reward'] for i in results]

def cn_value_align_reward_v2(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Chinese Value (STRICT CHECK VERSION)"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/cn_all_strict3_cluster.md"

    score_method = score_lastline

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def iraq_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Iraqi Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/iraq_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def german_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the German Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/german/german_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def korean_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Korean Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/korean/korean_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def turkey_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Turkish Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/turkey/turkey_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def bangladesh_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Bangladeshi Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/bangladesh/bangladesh_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def brazil_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Brazilian Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/brazil/brazil_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def mexico_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Mexican Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/mexico/mexico_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def argentina_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Argentinian Value"""
    
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/argentina/argentina_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def usa_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the US Value"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/usa_all_strict3_cluster.md"

    if "cluster" in prompt_path:
        score_method = score_lastline
    else:
        raise ValueError("Invalid prompt path")
    
    results = batch_score_responses(completion_contents, prompt_path, score_func=score_method)
    return [i['reward'] for i in results]

def all_mix_value_align_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of mixed all countries"""
    country = kwargs.get("country", [])[0]
    if country == "CHN":
        method = cn_value_align_reward_v2
    elif country == "USA":
        method = usa_value_align_reward
    elif country == "IRQ":
        method = iraq_value_align_reward
    elif country == "DEU":
        method = german_value_align_reward
    elif country == "KOR":
        method = korean_value_align_reward
    elif country == "TUR":
        method = turkey_value_align_reward
    elif country == "BGD":
        method = bangladesh_value_align_reward
    elif country == "BRA":
        method = brazil_value_align_reward
    elif country == "ARG":
        method = argentina_value_align_reward
    else:
        raise ValueError(f"Country not found: {country}")
    return method(completions, **kwargs)

def ablation_wo_cluster_cn(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of China ALL RULE"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]
    
    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/cn_all_strict3_allRule.md"

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_lastline)
    return [i['reward'] for i in results]

def ablation_wo_cluster_usa(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of American ALL RULE"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]

    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/usa_all_strict3_allRule.md"

    results = batch_score_responses(completion_contents, prompt_path, score_func=score_lastline)
    return [i['reward'] for i in results]

def ablation_wo_cluster_iraq(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of Iraq ALL RULE"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]

    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/iraq_all_strict3_allRule.md"
    results = batch_score_responses(completion_contents, prompt_path, score_func=score_lastline)
    return [i['reward'] for i in results]

def ablation_wo_cluster_brazil(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of Brazil ALL RULE"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]

    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/brazil/brazil_all_strict3_allRule.md"
    results = batch_score_responses(completion_contents, prompt_path, score_func=score_lastline)
    return [i['reward'] for i in results]


def ablation_wo_cluster_turkey(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of Turkish ALL RULE"""
    questions = kwargs.get("question")
    completion_contents = [{"answer": completion[0]["content"], "question": prompt} for completion, prompt in zip(completions, questions)]

    prompt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/turkey/turkey_all_strict3_allRule.md"
    results = batch_score_responses(completion_contents, prompt_path, score_func=score_lastline)
    return [i['reward'] for i in results]

def ablation_search_cn(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of China embedding search top5 Rule"""
    questions = kwargs.get("question")
    rules = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(kwargs.get("Search5CHN")[0]))
    completions = [{"answer": completion[0]["content"], "question": prompt, "rules": rules} for completion, prompt in zip(completions, questions)]

    promopt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/cn_all_strict3_search5top.md"
    results = batch_score_responses(completions, promopt_path, score_func=score_lastline)

    return [i['reward'] for i in results]

def ablation_search_iraq(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of Iraq embedding search top5 Rule"""
    questions = kwargs.get("question")
    rules = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(kwargs.get("Search5IRQ")[0]))
    completions = [{"answer": completion[0]["content"], "question": prompt, "rules": rules} for completion, prompt in zip(completions, questions)]

    promopt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/iraq_all_strict3_search5top.md"
    results = batch_score_responses(completions, promopt_path, score_func=score_lastline)

    return [i['reward'] for i in results]

def ablation_search_usa(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of USA embedding search top5 Rule"""
    questions = kwargs.get("question")
    rules = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(kwargs.get("Search5USA")[0]))
    completions = [{"answer": completion[0]["content"], "question": prompt, "rules": rules} for completion, prompt in zip(completions, questions)]

    promopt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/usa_all_strict3_search5top.md"
    results = batch_score_responses(completions, promopt_path, score_func=score_lastline)

    return [i['reward'] for i in results]

def ablation_search_turkey(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is align with the Value of Turkey embedding search top5 Rule"""
    questions = kwargs.get("question")
    rules = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(kwargs.get("Search5TUR")[0]))
    completions = [{"answer": completion[0]["content"], "question": prompt, "rules": rules} for completion, prompt in zip(completions, questions)]

    promopt_path = "/home/hzli/code/WVS/cultureSPA/scorePrompt/turkey/turkey_all_strict3_search5top.md"
    results = batch_score_responses(completions, promopt_path, score_func=score_lastline)

    return [i['reward'] for i in results]

def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        # "cn_value_align": cn_value_align_reward,
        "cn_value_align_v2": cn_value_align_reward_v2,
        "iraq_value_align": iraq_value_align_reward,
        "usa_value_align": usa_value_align_reward,
        "brazil_value_align": brazil_value_align_reward,
        "turkey_value_align": turkey_value_align_reward,
        "korean_value_align": korean_value_align_reward,
        "mexico_value_align": mexico_value_align_reward,
        "argentina_value_align": argentina_value_align_reward,
        "bangladesh_value_align": bangladesh_value_align_reward,
        "german_value_align": german_value_align_reward,
        "all_mix_value_align": all_mix_value_align_reward,
        "ablation_wo_cluster_iraq": ablation_wo_cluster_iraq,
        "ablation_wo_cluster_usa": ablation_wo_cluster_usa,
        "ablation_wo_cluster_cn": ablation_wo_cluster_cn,
        "ablation_wo_cluster_brazil": ablation_wo_cluster_brazil,
        "ablation_wo_cluster_turkey": ablation_wo_cluster_turkey,
        "ablation_search_cn": ablation_search_cn,
        "ablation_search_iraq": ablation_search_iraq,
        "ablation_search_usa": ablation_search_usa,
        "ablation_search_turkey": ablation_search_turkey,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
