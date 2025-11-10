import re
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_json(path):
    if path.endswith('.jsonl'):
        return [json.loads(line) for line in open(path, 'r')]
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, s):
    with open(path, 'w') as f:
        json.dump(s, f, indent=4, ensure_ascii=False)

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

def score(args):
    resp, prompt = args
    score_resp = get_response(prompt, model_name="gpt-4o-mini")
    try:
        num = re.findall(r"[01234]", score_resp.split('\n')[0])[-1]
    except:
        num = 0
    # print(num)
    return {
        "question": resp["question"],
        "answer": resp["answer"],
        "score": score_resp, 
        "reward": int(num),
        "from": resp["from"]
    }

def batch_score_responses(responses, prompt_path, max_workers=10):
    from tqdm import tqdm 
    prompt_prefix = load_prompt(prompt_path)
    results = [None] * len(responses)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(score, (resp, prompt_prefix + f"Question: {resp['question']}\nResponse: "+ resp["answer"])): idx
            for idx, resp in enumerate(responses)
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Scoring"):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"Error: {str(e)}"
    # print(results)
    return results

def cn_value_align_reward(infer_data):
    """Reward function that checks if the completion is align with the Chinese Value"""
    
    completion_contents = [{"answer": completion["generated_text"], "question": completion["question"], "from": completion["from"]} for completion in infer_data]
    results = batch_score_responses(completion_contents, "cultureSPA/scorePrompt/cn_all_strict.md")
    def form(reward):
        if reward <= 0:
            return reward
        return (reward) / 3
    return results

def calc(metric_data):
    # metric_data = read_json(file_path)
    s = sum([int(i["reward"]) for i in metric_data])
    return s / len(metric_data)

def main(FILE_PATH):
    import os
    import datetime
    date = datetime.datetime.now().strftime("%m-%d_%H-%M")
    
    data = read_json(FILE_PATH)
    results = cn_value_align_reward(data)

    parent_directory = os.path.dirname(FILE_PATH)
    metric_folder_path = os.path.join(parent_directory, "metric_reward_func")
    if os.path.exists(metric_folder_path) == False:
        os.makedirs(metric_folder_path)

    write_json(os.path.join(metric_folder_path, f"reward_score_{date}_{calc(results)}.json"), results)

if __name__ == "__main__":
    import fire
    fire.Fire(main)