import json
import os
import fire
from sklearn.metrics import f1_score, accuracy_score
import re
from collections import defaultdict

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

def postprocess(task, output):
    """
    将模型原始输出的文本转换为标准化的标签。
    """
    if output is None or output == '':
        return "" # 返回一个空字符串，避免后续处理出错
    
    label = "" # 默认标签
    
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        if 'not' in output.lower():
            label = 'NOT'
        elif 'off' in output.lower():
            label = 'OFF'
        else:
            label = output
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        if 'not' in output.lower():
            label = 'NOT_HS'
        elif 'hs' in output.lower():
            label = 'HS'
        else:
            label = output
    elif task == 'vulgar_detect_mp':
        if 'not' in output.lower():
            label = '-'  
        elif 'vulgar' in output.lower():
            label = 'V'
        else:
            label = output
    elif task == 'spam_detect':
        if 'no' in output.lower() or 'not' in output.lower(): # 增加对'not'的判断
            label = 'Ham'
        elif 'spam' in output.lower():
            label = 'Spam'
        else:
            label = output
    elif task == 'hate_detect_fine-grained':
        if 'not' in output.lower(): label = 'NOT_HS'
        elif '1' in output: label = 'HS1'
        elif '2' in output: label = 'HS2'
        elif '3' in output: label = 'HS3'
        elif '4' in output: label = 'HS4'
        elif '5' in output: label = 'HS5'
        elif '6' in output: label = 'HS6'
        else: label = output
    elif task == 'offensive_detect_finegrained':
        if 'no' in output.lower(): label = 'non'
        elif 'prof' in output.lower(): label = 'prof'
        elif 'grp' in output.lower(): label = 'grp'
        elif 'indv' in output.lower(): label = 'indv'
        elif 'oth' in output.lower(): label = 'oth'
        else: label = output
    elif task == 'hate_off_detect':
        if 'no' in output.lower():
            label = 'NOT'
        elif 'hof' in output.lower():
            label = 'HOF'
        else:
            label = output
    elif task in ['stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect', 'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect', 'threat_detect', 'hostility_directness_detect']:
        if '1' in output:
            label = '1'
        elif '0' in output:
            label = '0'
        else:
            label = output
    elif task == 'hate_offens_detect':
        if '0' in output: label = '0'
        elif '1' in output: label = '1'
        else: label = '2' # 默认为else类别
    elif task == 'GSM8K':
        pred = str(output).lower()
        pred = re.sub(r"[,']", "", pred) # 移除逗号和引号
        numbers = re.findall(r"\d+\.?\d*", pred) # 查找数字（包括小数）
        if numbers:
            label = numbers[-1]
        else:
            label = "" # 如果没找到数字，返回空
    else:
        label = output # 如果没有匹配的任务，返回原始输出

    return label

def computeMetrics(task, gt_list, pred_list):
    """
    根据任务类型，计算F1分数或准确率。
    """
    final_f1_score = 0.0 # 默认值
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        final_f1_score = f1_score(gt_list, pred_list, labels=['OFF', 'NOT'], average='macro', zero_division=0)
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['HS', 'NOT_HS'], average='macro', zero_division=0)
    elif task == 'vulgar_detect_mp':
        final_f1_score = f1_score(gt_list, pred_list, labels=['V', '-'], average='macro', zero_division=0)
    elif task == 'spam_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['Spam', 'Ham'], average='macro', zero_division=0)
    elif task == 'hate_detect_fine-grained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['NOT_HS', 'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6'], average='macro', zero_division=0)
    elif task == 'offensive_detect_finegrained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['non', 'prof', 'grp', 'indv', 'oth'], average='macro', zero_division=0)
    elif task == 'hate_off_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['HOF', 'NOT'], average='macro', zero_division=0)
    elif task in ['stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect', 'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect', 'threat_detect', 'hostility_directness_detect']:
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro', zero_division=0)
    elif task == 'hate_offens_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1', '2'], average='macro', zero_division=0)
    elif task == 'GSM8K':
        final_f1_score = accuracy_score(gt_list, pred_list)

    return final_f1_score


def calculate_score_for_file(json_file_path: str):
    """
    对单个推理结果JSON文件进行评分的核心逻辑。
    (此函数包含了您原来main函数的所有功能)
    """
    if not os.path.exists(json_file_path):
        print(f"  -> 警告: 结果文件未找到，跳过评分: {json_file_path}")
        return None

    taskname = os.path.basename(json_file_path).replace('.json', '')
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  -> 错误: 读取或解析JSON文件 '{json_file_path}' 时出错: {e}")
        return None

    gt_list, pred_list = [], []
    for item in data:
        if 'label' not in item or 'generated_text' not in item:
            continue
        gt_list.append(str(item['label']))
        pred_list.append(str(postprocess(taskname, item['generated_text'])))

    if not gt_list:
        print(f"  -> 警告: 文件 '{json_file_path}' 中未找到可评分的有效项目。")
        return None

    try:
        score = computeMetrics(taskname, gt_list, pred_list)
        metric_name = "Accuracy" if taskname == 'GSM8K' else "Macro F1-Score"
        print(f"  -> 任务 '{taskname}' 评分完成. 指标: {metric_name}, 得分: {score:.4f}")
        return score
    except Exception as e:
        print(f"  -> 错误: 为任务 '{taskname}' 计算指标时发生错误: {e}")
        return None


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    if data is None:
        data = {}
    return data

def main(inference_dir: str, language: str):
    """
    自动评估指定语言下的所有任务，并生成两种格式的结果文件：
    1. all_results.json (字典格式)
    2. all_results_horizontal.tsv (横向表格格式)
    """
    print(f"--- 开始为语言 '{language}' 在目录 '{inference_dir}' 下的所有任务进行批量评分 ---")

    lang_key = language.lower()
    tasks_to_run = list(DATA_MAP.get(lang_key, {}).keys())

    if not tasks_to_run:
        print(f"错误: 在DATA_MAP中未找到为语言 '{language}' 定义的任何任务。")
        return

    print(f"检测到 {len(tasks_to_run)} 个任务: {tasks_to_run}")
    
    all_results = {}

    for taskname in tasks_to_run:
        json_file_path = os.path.join(inference_dir, f"{taskname}.json")
        score = calculate_score_for_file(json_file_path)
        if score is not None:
            all_results[taskname] = score

    # --- 核心修改部分：生成并保存两种格式的结果 ---
    if not all_results:
        print("\n所有任务均未成功评分，未生成结果文件。")
        return
        
    print("\n--- 所有任务评分完毕，正在生成结果文件 ---")
    
    # 1. 保存原始的 all_results.json 文件
    json_output_path = os.path.join(inference_dir, "all_results.json")

    if os.path.exists(json_output_path):
        final_x = read_json(json_output_path)
        final_x.update(all_results)
        all_results = final_x
        # print(all_results)
        # print(read_json(json_output_path))
        # i = input()

    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"字典格式的结果已保存到: {json_output_path}")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")

    # 2. 生成并保存横向的 all_results_horizontal.tsv 文件
    tsv_output_path = os.path.join(inference_dir, "all_results_horizontal.tsv")
    try:
        # 为了保证每次的列顺序一致，我们对任务名进行排序
        sorted_tasks = sorted(all_results.keys())
        
        # 创建表头行 (所有任务名，用Tab分隔)
        header_row = "\t".join(sorted_tasks)
        
        # 创建分数行 (所有分数，按同样顺序，用Tab分隔)
        score_row = "\t".join(f"{all_results[task]:.4f}" for task in sorted_tasks)
        
        # 将两行用换行符连接
        tsv_content = f"{header_row}\n{score_row}"
        
        with open(tsv_output_path, 'w', encoding='utf-8') as f:
            f.write(tsv_content)
        print(f"横向表格格式的结果已保存到: {tsv_output_path}")
        
        # 在屏幕上也打印出来，方便直接复制
        print("\n--- 横向结果 (可直接复制到表格) ---")
        print(tsv_content)
        print("-" * 50)
        
    except Exception as e:
        print(f"保存TSV文件时出错: {e}")



if __name__ == '__main__':
    # 请确保您已将完整的DATA_MAP和postprocess/computeMetrics函数逻辑粘贴到此脚本中
    fire.Fire(main)