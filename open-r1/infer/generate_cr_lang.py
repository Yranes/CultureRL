import json
import os
import fire
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM or Transformers not found. Running in API-only mode.")

import jsonlines


DATA_MAP = {
    "arabic": {
        "offensive_detect": "data/Arabic/OffensEval2020/OffensEval.jsonl",
        "hate_detect_osact4": "data/Arabic/OSACT4/dev_data.jsonl",
        "offensive_detect_osact4": "data/Arabic/OSACT4/dev_data_offens.jsonl",
        "offensive_detect_mp": "data/Arabic/MP/offens.jsonl",
        "hate_detect_mp": "data/Arabic/MP/hateSpeech.jsonl",
        "vulgar_detect_mp": "data/Arabic/MP/VulgarSpeech.jsonl",
        "spam_detect": "data/Arabic/SpamDetect/span_detect_2.jsonl",
        "offensive_detect_osact5": "data/Arabic/OSACT5/offens.jsonl",
        "hate_detect_osact5": "data/Arabic/OSACT5/hateSpeech.jsonl",
        "hate_detect_fine-grained": "data/Arabic/OSACT5/hate_Finegrained.jsonl"
    },
    "turkish": {
        "offensive_detect": "data/Turkey/OffensEval2020/OffensEval.jsonl",
        "offensive_detect_corpus": "data/Turkey/offenseCorpus/offens.jsonl",
        "offensive_detect_finegrained": "data/Turkey/offenseCorpus/offens_fine-graind.jsonl",
        "offensive_detect_kaggle": "data/Turkey/offenssDetect-kaggle/turkish_tweets_2020.jsonl",
        "offensive_detect_kaggle2": "data/Turkey/offensDetect-kaggle2/test.jsonl",
        "abusive_detect": "data/Turkey/ATC/fold_0_test.jsonl",
        "spam_detect": "data/Turkey/TurkishSpam/trspam.jsonl"
    },
    "greek": {
        "offensive_detect": "data/Greece/OffensEval2020/OffensEval.jsonl",
        "offensive_detect_g": "data/Greece/gazzetta/G-TEST-S-preprocessed.jsonl"
    },
    "german": {
        "hate_detect": "data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_2.jsonl",
        "hate_off_detect": "data/Germany/HASOC/hate_off_detect.jsonl",
        "offensive_detect_eval": "data/Germany/GermEval/germeval2018.jsonl",
        "hate_detect_check": "data/Germany/MHC/hatecheck_cases_final_german.jsonl",
        "hate_detect_iwg_1": "data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_1.jsonl"
    },
    "spanish": {
        "offensive_detect_ami": "data/Spanish/AMI IberEval 2018_offens/data-2.jsonl",
        "offensive_detect_mex_a3t": "data/Spanish/MEX-A3T_offens/data-2.jsonl",
        "offensive_detect_mex_offend": "data/Spanish/OffendES_offens/data-2.jsonl",
        "hate_detect_eval": "data/Spanish/HateEval 2019_HS/data-2.jsonl",
        "hate_detect_haterNet": "data/Spanish/HaterNet_HS/data-2.jsonl",
        "stereotype_detect": "data/Spanish/DETOXIS 2021/stereotype.jsonl",
        "mockery_detect": "data/Spanish/DETOXIS 2021/mockery.jsonl",
        "insult_detect": "data/Spanish/DETOXIS 2021/insult.jsonl",
        "improper_detect": "data/Spanish/DETOXIS 2021/improper_language.jsonl",
        "aggressiveness_detect": "data/Spanish/DETOXIS 2021/aggressiveness.jsonl",
        "negative_stance_detect": "data/Spanish/DETOXIS 2021/negative_stance.jsonl"
    },
    "bengali": {
        "offensive_detect_1": "data/Bengali/Trac2-Task1-Aggresion/aggression-data-2.jsonl",
        "offensive_detect_2": "data/Bengali/Trac2-Task2-Misogynistic/Misogynistic-data-2.jsonl",
        "offensive_detect_3": "data/Bengali/BAD-Bangla-Aggressive-Text-Dataset/data-2.jsonl",
        "hate_detect_religion": "data/Bengali/Bengali hate speech dataset/religion_data-2.jsonl",
        "threat_detect": "data/Bengali/Bangla-Abusive-Comment-Dataset/threat.jsonl",
        "racism_detect": "data/Bengali/Bangla-Abusive-Comment-Dataset/racism.jsonl"
    },
    "chinese": {
        "spam_detect": "data/China/Chinese-Camouflage-Spam-dataset/data-2.jsonl",
        "bias_on_gender_detect": "data/China/CDial-Bias/gender-2.jsonl"
    },
    "korean": {
        "hate_detect_3": "data/Korean/K-MHaS/data-2.jsonl",
        "hate_detect_6": "data/Korean/Korean-Hate-Speech-Detection/data-2.jsonl",
        "hate_detect_7": "data/Korean/KoreanHateSpeechdataset/data-2.jsonl",
        "abusive_detect": "data/Korean/AbuseEval/data-2.jsonl",
        "abusive_detect_2": "data/Korean/CADD/data-2.jsonl",
        "abusive_detect_4": "data/Korean/Waseem/data-2.jsonl"
    },
    "portuguese": {
        "offensive_detect_2": "data/Portuguese/OffComBR/data.jsonl",
        "offensive_detect_3": "data/Portuguese/HateBR/data-2.jsonl",
        "homophobia_detect": "data/Portuguese/ToLD-Br/homophobia.jsonl",
        "misogyny_detect": "data/Portuguese/ToLD-Br/misogyny.jsonl",
        "insult_detect": "data/Portuguese/ToLD-Br/insult.jsonl"
    },
    "english": {
        "hate_detect_2": "data/English/MLMA hate speech/data-2.jsonl",
        "hostility_directness_detect": "data/English/MLMA hate speech/directness.jsonl",
        "hate_offens_detect": "data/English/hate-speech-and-offensive-language/data.jsonl",
        "offensive_detect_easy": "data/English/SOLID/test_a_tweets_easy.jsonl",
        "toxicity_detect": "data/English/Toxic Comment Classification Challenge/toxic.jsonl",
        "threat_detect": "data/English/Toxic Comment Classification Challenge/threat.jsonl"
    }
}


def getPrompt(input, task, input_1=None):
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        prompt = f"if the following sentence is offensive, just say \"OFF\", otherwise, say \"NOT_OFF\" without explanation: {input}"
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        prompt = f"if the following sentence has hate speech, just say \"HS\", otherwise, say \"NOT_HS\" without explanation: {input}"
    elif task == 'vulgar_detect_mp':
        prompt = f"if the following sentence has vulgar speech, just say \"Vulgar\", otherwise, say \"NOT_Vulgar\" without explanation: {input}"
    elif task == 'spam_detect':
        prompt = f"Is the following tweet spam? Answer with only \"Spam\" or \"NOT_Spam\".Tweet: \"{input}\""
    elif task == 'hate_detect_fine-grained':
        prompt = f"if the following sentence doesn't have hate speech, just say \"NOT_HS\", otherwise, label the hate speech with \"HS1\"(Race), \"HS2\"(Religion), \"HS3\"(Ideology), \"HS4\"(Disability), \"HS5\"(Social Class), \"HS6\"(Gender) without explanation: {input}"
    elif task == 'offensive_detect_finegrained':
        prompt = f"if the following sentence doesn't have offensive speech, just say \"non\", otherwise, label the offensive speech with \"prof\"(profanity, or non-targeted offense), \"grp\"(offense towards a group), \"indv\"(offense towards an individual), \"oth\"(ffense towards an other (non-human) entity, often an event or organization) without explanation: {input}"
    elif task == 'hate_off_detect':
        prompt = f"if the following sentence has hate speech or offensive content, just say \"HOF\", otherwise, say \"NOT\" without explanation: {input}"
    elif task == 'stereotype_detect' or task == 'mockery_detect' or task == 'insult_detect' or task == 'improper_detect' or task == 'aggressiveness_detect' or task == 'toxicity_detect' or task == 'negative_stance_detect' or task == 'homophobia_detect' or task == 'racism_detect' or task == 'misogyny_detect' or task == 'threat_detect':
        entity = task[:-7]
        prompt = f"if the following sentence has {entity} speech, just say \"1\", otherwise, say \"0\" without explanation: {input}"
    elif task == 'bias_on_gender_detect' or task == 'hostility_directness_detect':
        entity = task[:-7]
        entity = entity.replace('_', ' ')
        prompt = f"if the following speech expressing {entity}, just say \"1\", otherwise, say \"0\" without explanation: {input}"
    elif task == 'hate_offens_detect':
        prompt = f"if the following sentence contains hate speech, just say \"0\", else if contains offensive language, say \"1\", otherwise, say \"2\" without explanation: {input}"
    else:
        prompt = input
    return prompt

def getDataPath(language, task):
    """重构后的函数：从DATA_MAP中查找路径。"""
    base_path = 'open-r1/data/mod'
    lang_tasks = DATA_MAP.get(language.lower())
    # 兼容像GSM8K这样以任务名作为主键的特殊情况
    if not lang_tasks:
        lang_tasks = DATA_MAP.get(task.lower())
    
    if not lang_tasks:
        raise ValueError(f"Unsupported language or task group: {language}")

    path = lang_tasks.get(task)
    if path is None:
        raise ValueError(f"Unsupported task '{task}' for language '{language}'")
    
    return os.path.join(base_path, path)

def get_response(messages, model_name):
    """调用API并返回(content, error)元组"""
    client = openai.OpenAI(
        api_key="sk-X", 
        base_url="",
    )
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )
        return (res.choices[0].message.content.strip(), None)
    except Exception as e:
        return (None, f"Error: {str(e)}")

l2c = {
    "arabic": "Iraqi",
    "english": "American",
    "chinese": "Chinese",
    "bengali": "Bangladeshi",
    "portuguese": "Brazilian",
    "german": "German",
    "korean": "Korean",
    "turkish": "Turkish",
    "spanish": "Argentinian",
}

def run_api_task(ModelPath, language, taskname, max_workers=20):
    """为单个任务执行API推理和保存。"""
    print(f"\n--- Running API task: '{taskname}' for language '{language}' ---")
    country = l2c.get(language.lower())
    # 加载数据和准备messages的逻辑与之前相同
    jsonl_path = getDataPath(language, taskname)
    original_data = [item for item in jsonlines.Reader(open(jsonl_path, "r", encoding="utf8"))]
    
    messages_list = []
    for item in original_data:
        input_text = item.get('comment') or item.get('data') or item.get('tweet') or item.get('question')
        if input_text is None: continue
        prompt = getPrompt(input_text, taskname)
        messages = [{"role": "system", "content": f"You are a real person with a {country} cultural background. Please answer the questions honestly according to your own culture value system."}, {"role": "user", "content": prompt}]
        messages_list.append(messages)
        
    print("example: ", messages_list[0])
    results = [item for item in original_data]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(get_response, msgs, ModelPath): i for i, msgs in enumerate(messages_list)}
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"API - {taskname}"):
            idx = future_to_index[future]
            try:
                generated_text, error = future.result()
                results[idx]['generated_text'] = error if error else generated_text
            except Exception as e:
                results[idx]['generated_text'] = f"Future Error: {str(e)}"
    
    # 保存逻辑
    OUTPUT_DIR = f"open-r1/infer/"
    OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, ModelPath.replace("/", "_"), language)
    output_path = os.path.join(OUTPUT_FOLDER, f"{taskname}.json")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Saving results for '{taskname}' to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Task '{taskname}' finished successfully.")


def run_vllm_task(llm, tokenizer, ModelPath, language, taskname):
    """
    为单个任务执行VLLM推理和保存。
    注意：此函数接收已加载的llm和tokenizer对象。
    """
    print("\n" + "="*80)
    print(f"Executing task: '{taskname}' for language: '{language}' using pre-loaded VLLM model.")
    print("="*80)

    country = l2c[language.lower()]

    # 加载数据和准备prompts的逻辑与之前相同
    jsonl_path = getDataPath(language, taskname)
    original_data = [item for item in jsonlines.Reader(open(jsonl_path, "r", encoding="utf8"))]

    prompts = []
    for item in original_data:
        input_text = item.get('comment') or item.get('data') or item.get('tweet') or item.get('question')
        if input_text is None: continue
        prompt = getPrompt(input_text, taskname)
        prompts.append(tokenizer.apply_chat_template([{"role": "system", "content": f"You are a real person with a {country} cultural background. Please answer the questions honestly according to your own culture value system."},{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))

    # 执行推理
    sampling_params = SamplingParams(max_tokens=16, temperature=0.6, top_p=0.95)
    vllm_results = llm.generate(prompts, sampling_params)
    
    # 组装和保存结果
    results_list = []
    for i, output in enumerate(vllm_results):
        data_item = original_data[i]
        data_item['prompt'] = output.prompt
        data_item['generated_text'] = output.outputs[0].text.strip()
        results_list.append(data_item)
        
    # 定义输出路径的逻辑保持不变
    OUTPUT_DIR = "/home/hzli/code/WVS/open-r1/infer/"
    path_parts = ModelPath.strip("/").split('/')
    num = -1
    while num >= -len(path_parts):
        part = path_parts[num]
        if "train_" in part or "Qwen2.5-" in part: break
        num -= 1
    else: num = -1
    
    OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, path_parts[num])
    
    if "Qwen2.5-3B-Instruct" in ModelPath:
        OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, language)
    
    if path_parts[-1].startswith("checkpoint"):
        OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, path_parts[-1])
    
    output_path = os.path.join(OUTPUT_FOLDER, f"{taskname}.json")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Results for '{taskname}' will be saved to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)
    print(f"Task '{taskname}' finished successfully.")

# --- 4. 主函数 (任务分发器) ---
def main(ModelPath, language, taskname=None, max_workers=16):
    """
    主函数，负责加载模型（如果需要）并分发任务。
    """
    # 确定要运行的任务列表
    if taskname:
        tasks_to_run = [taskname]
    else:
        lang_key = language.lower()
        tasks_to_run = list(DATA_MAP.get(lang_key, {}).keys())
        if not tasks_to_run:
            print(f"Error: No tasks found for language '{lang_key}'. Please check DATA_MAP.")
            return
    
    print(f"Found {len(tasks_to_run)} task(s) to run for language '{language}': {tasks_to_run}")

    if "gpt" in ModelPath.lower():
        for task in tasks_to_run:
            run_api_task(ModelPath, language, task, max_workers)
    else:
        if not VLLM_AVAILABLE:
            print("Error: VLLM libraries are not installed. Cannot run local models.")
            return
            
        print(f"\nInitializing VLLM with model: {ModelPath} (This will happen only once)")
        llm = LLM(model=ModelPath, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ModelPath, trust_remote_code=True)
        print("VLLM model loaded successfully.")

        for task in tasks_to_run:
            run_vllm_task(llm, tokenizer, ModelPath, language, task)
    
    print("\nAll specified tasks have been completed.")


if __name__ == '__main__':
    fire.Fire(main)