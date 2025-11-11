import json
import os
import fire
from sklearn.metrics import f1_score, accuracy_score
import re

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

def score_predictions(json_file_path: str):
    """
    对VLLM生成的JSON结果文件进行评分。

    Args:
        json_file_path (str): 推理脚本生成的JSON文件的路径。
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件未找到于 {json_file_path}")
        return

    # 从文件名推断任务名称
    # 假设文件名格式为 '任务名.json' 或 '任务名_模型名.json'
    # 我们取第一个下划线前的部分作为任务名
    filename_base = os.path.basename(json_file_path).split('.')[0]
    taskname = filename_base

    print(f"正在处理文件: {json_file_path}")
    print(f"检测到任务名称为: {taskname}")

    # 加载JSON数据
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取或解析JSON文件时出错: {e}")
        return

    # 准备真实标签和预测标签列表
    gt_list = []
    pred_list = []

    for item in data:
        # 确保必需的键存在
        if 'label' not in item or 'generated_text' not in item:
            print(f"警告: 跳过一个项目，因其缺少'label'或'generated_text'键: {item}")
            continue

        gt_label = item['label']
        raw_prediction = item['generated_text']

        # 后处理模型输出以获得干净的标签
        clean_prediction = postprocess(taskname, raw_prediction)

        gt_list.append(str(gt_label))  # 确保标签为字符串以便比较
        pred_list.append(str(clean_prediction))

    if not gt_list:
        print("未找到可评分的有效项目。")
        return

    # 计算最终得分
    print(f"正在为 {len(gt_list)} 个项目计分...")
    try:
        final_score = computeMetrics(taskname, gt_list, pred_list)
        metric_name = "Accuracy" if taskname == 'GSM8K' else "Macro F1-Score"
        
        # 打印结果
        print("\n" + "="*40)
        print(f"评分结果 - {filename_base}")
        print(f"任务 (Task): {taskname}")
        print(f"评估指标 (Metric): {metric_name}")
        print(f"最终得分 (Final Score): {final_score:.4f}")
        print("="*40 + "\n")

    except Exception as e:
        print(f"在计算指标时发生错误: {e}")
        print("这可能是因为预测标签与预期标签不匹配。")
        print(f"真实标签集合: {sorted(list(set(gt_list)))}")
        print(f"预测标签集合: {sorted(list(set(pred_list)))}")


if __name__ == '__main__':
    fire.Fire(score_predictions)