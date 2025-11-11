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

suoxie2country = {
    "IRQ": "arabic",
    "BGD": "bengali",
    "BRA": "portuguese",
    "DEU": "german",
    "KOR": "korean",
    "TUR": "turkish",
    "ARG": "spanish",
}

def main(MODEL_PATH, JSONL_PATH, PEFT_PATH = None):
    # 路径配置
    # MODEL_PATH = "/home/share/models/Qwen2.5-3B-Instruct"
    # JSONL_PATH = "open-r1/data/test/test600_split_65.json"

    # 初始化 LLM
    if PEFT_PATH:
        print("PEFT_MODEL")
        llm = LLM(model=MODEL_PATH, enable_lora=True, max_lora_rank=64)
    else:
        print("FULL_MODEL")
        llm = LLM(model=MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载你的jsonl
    prompts = []
    test_data = read_json(JSONL_PATH)
    for line in test_data:
        prompt_msgs = line["prompt"]
        prompts.append(tokenizer.apply_chat_template(prompt_msgs, tokenize = False, add_generation_prompt = True))

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0,
        # top_p=0.95,
    )
    sampling_params_logits = SamplingParams(
        logprobs = 20
    )
    # === 直接推理 (多条batch，格式为chat消息列表) ===
    if PEFT_PATH:
        results = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest("wvs_rule_chinese", 1, PEFT_PATH)
        )
        results_logit = llm.generate(
            prompts,
            sampling_params_logits,
            lora_request=LoRARequest("wvs_rule_chinese", 1, PEFT_PATH)
        )
    else:
        results = llm.generate(
            prompts,
            sampling_params,
        )
        results_logit = llm.generate(
            prompts,
            sampling_params_logits,
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
    if PEFT_PATH:
        OUTPUT_DIR = "open-r1/infer/"
        num = -1
        while "train_" not in PEFT_PATH.split('/')[num]:
            num -= 1
        OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{PEFT_PATH.split('/')[num]}")
        if PEFT_PATH.split('/')[-1].startswith("checkpoint"):
            OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{PEFT_PATH.split('/')[num]}", f"{PEFT_PATH.split('/')[-1]}")
        OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{JSONL_PATH.split('/')[-1].split('.')[0]}.json")
    else:
        OUTPUT_DIR = "open-r1/infer/"
        num = -1
        while "train_" not in MODEL_PATH.split('/')[num] and "Qwen2.5-" not in MODEL_PATH.split('/')[num]:
            num -= 1
        OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{MODEL_PATH.split('/')[num]}")
        if MODEL_PATH.split('/')[-1].startswith("checkpoint"):
            OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, f"{MODEL_PATH.split('/')[num]}", f"{MODEL_PATH.split('/')[-1]}")
        if "Qwen2.5-3B-Instruct" in MODEL_PATH:
            OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, suoxie2country[JSONL_PATH.split('/')[-1].split('.')[0].split('_')[-1]].lower())
        
        OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{JSONL_PATH.split('/')[-1].split('.')[0]}.json")

    if os.path.exists(OUTPUT_FOLDER) == False:
        os.makedirs(OUTPUT_FOLDER)
    
    vllmwrite = test_data
    for i, (output, output_logprob) in enumerate(zip(results, results_logit)):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(output_logprob.outputs[0].logprobs)
        vllmwrite[i].update({
            "prompt_input": prompt,
            "question": prompt.split("<|im_start|>user\n")[1].split("<|im_end|>\n")[0].strip("\n"),
            "generated_text": generated_text,
            "logprob": {i.decoded_token: i.logprob for i in output_logprob.outputs[0].logprobs[0].values()},
        })
    print(OUTPUT_PATH)
    write_json(OUTPUT_PATH, vllmwrite)
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)