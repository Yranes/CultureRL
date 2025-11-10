from vllm import LLM, SamplingParams
import json
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer

def write_json(path, s):
    with open(path, "w") as f:
        json.dump(s, f, indent=4, ensure_ascii=False)

def read_json(path):
    if path.endswith('.jsonl'):
        return [json.loads(line) for line in open(path, 'r')]
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def vllmoutput2json(vllmout):
    results = []
    for output in vllmout:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append({
            "prompt_input": prompt,
            "question": prompt.split("<|im_start|>user\n")[1].split("<|im_end|>\n")[0].strip("\n"),
            "generated_text": generated_text,
        })
    return results

def main(MODEL_PATH, JSONL_PATH, country="Chinese"):

    # 初始化 LLM
    llm = LLM(model=MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载你的jsonl
    prompts = []
    test_data = read_json(JSONL_PATH)
    for line in test_data:
        if line.get("prompt") is not None:
            prompt_msgs = line["prompt"]
        elif line.get("question") is not None:
            prompt_msgs = [
                {"role": "system", "content": f"You are a real person with a {country} cultural background."},
                {"role": "user", "content": line["question"]}
            ]
        else:
            raise ValueError("Key Error: prompt or question is required")
        prompts.append(tokenizer.apply_chat_template(prompt_msgs, tokenize = False, add_generation_prompt = True))

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.76,
        top_p=0.9,
    )

    # === 直接推理 (多条batch，格式为chat消息列表) ===
    results = llm.generate(
        prompts,
        sampling_params,
    )

    print("infer ok")
    # 保存或打印
    example = results[0]
    prompt = example.prompt
    generated_text = example.outputs[0].text
    print(f"Example - Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # import datetime
    # datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    import os
    OUTPUT_DIR = "open-r1/infer/"
    num = -1
    while "train_" not in MODEL_PATH.split('/')[num] and "Qwen2.5-" not in MODEL_PATH.split('/')[num]:
        num -= 1
    
    OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{MODEL_PATH.split('/')[num]}")

    if "Qwen2.5-3B-Instruct" in MODEL_PATH:
        OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, country.lower())

    country2suoxie = {
        "Chinese": "CHN",
        "American": "USA",
        "Iraqi": "IRQ",
        "Bangladeshi": "BGD",
        "Argentinian": "ARG",
        "German": "DEU",
        "Korean": "KOR",
        "Turkish": "TUR",
        "Brazilian": "BRA"
    }


    if MODEL_PATH.split('/')[-1].startswith("checkpoint"):
        OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{MODEL_PATH.split('/')[num]}", f"{MODEL_PATH.split('/')[-1]}")
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{JSONL_PATH.split('/')[-1].split('.')[0]}_{country2suoxie[country]}.json")

    if os.path.exists(OUTPUT_FOLDER) == False:
        os.makedirs(OUTPUT_FOLDER)
    
    vllmwrite = test_data
    for i, output in enumerate(results):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(output_logprob.outputs[0].logprobs)
        vllmwrite[i].update({
            "prompt_input": prompt,
            "question": prompt.split("<|im_start|>user\n")[1].split("<|im_end|>\n")[0].strip("\n"),
            "generated_text": generated_text
        })
    print(OUTPUT_PATH)
    write_json(OUTPUT_PATH, vllmwrite)
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)