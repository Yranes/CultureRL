import os
import json
import csv
import re

def find_all_results(base_dir):
    """查找所有符合条件的all_results.json文件路径。"""
    result_files = []
    if not os.path.isdir(base_dir):
        print(f"错误: 目录不存在 -> {base_dir}")
        return result_files

    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-"):
            ckpt_path = os.path.join(base_dir, item)
            json_path = os.path.join(ckpt_path, "all_results.json")
            if os.path.isfile(json_path):
                result_files.append(json_path)
    return result_files

def get_ckpt_number(path):
    """从路径中提取checkpoint的数字部分，用于排序。"""
    ckpt_name = os.path.basename(os.path.dirname(path))
    match = re.search(r'\d+', ckpt_name)
    return int(match.group(0)) if match else 0

def aggregate_and_create_csv(infer_base_path):
    """
    主函数：扫描、聚合数据、计算平均分并生成最终的CSV文件。
    """
    output_csv_path = os.path.join(infer_base_path, "excel_scores.csv")
    print(f"正在扫描目录: {infer_base_path}")
    json_files = find_all_results(infer_base_path)

    if not json_files:
        print("未找到任何 'all_results.json' 文件。")
        return

    json_files.sort(key=get_ckpt_number)

    all_data = []
    all_headers = set()

    # --- 1. 读取所有数据并收集所有表头 ---
    for f_path in json_files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ckpt_name = os.path.basename(os.path.dirname(f_path))
                data['ckptname'] = ckpt_name
                all_data.append(data)
                all_headers.update(data.keys())
        except Exception as e:
            print(f"读取或处理文件时出错: {f_path}, 错误: {e}")
    
    if not all_data:
        print("未能从任何文件中成功加载数据。")
        return

    # --- 2. 按照您的要求对表头进行排序 ---
    special_headers = ['ckptname', '1-JSD']
    vsm_headers = sorted([h for h in all_headers if h.startswith('vsm')])
    
    # 定义需要排除的列
    exclusion_list = special_headers + vsm_headers
    
    # 获取需要计算平均值的“内容审查任务”的表头
    content_mod_headers = sorted([
        h for h in all_headers if h not in exclusion_list
    ])
    
    # 组合最终表头，并将新列名放在最后
    final_ordered_headers = special_headers + vsm_headers + content_mod_headers
    final_ordered_headers.append('Content_Moderation_Average') # <-- 改动1: 新增平均值列头

    # --- 3. 格式化数据并计算平均值 ---
    formatted_data = []
    for row_dict in all_data:
        new_row = {}
        
        # --- 改动2: 计算平均值 ---
        scores_to_average = []
        for header in content_mod_headers:
            value = row_dict.get(header)
            # 确保值是数字再加入计算列表
            if isinstance(value, (int, float)):
                scores_to_average.append(value)
        
        # 计算平均分，如果列表不为空
        average_score = sum(scores_to_average) / len(scores_to_average) if scores_to_average else None
        # 将计算出的平均分添加到原始数据字典中，以便后续统一格式化
        row_dict['Content_Moderation_Average'] = average_score

        # 格式化所有数据，包括新计算的平均分
        for header in final_ordered_headers:
            value = row_dict.get(header)
            if isinstance(value, (int, float)):
                # --- 改动3: 将所有数值格式化为四位小数 ---
                new_row[header] = f"{value:.4f}"
            else:
                new_row[header] = value # 保持ckptname或None值不变
        formatted_data.append(new_row)


    # --- 4. 将格式化后的数据写入CSV文件 ---
    print(f"\n正在将 {len(formatted_data)} 个checkpoint的结果写入到: {output_csv_path}")
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=final_ordered_headers)
            writer.writeheader()
            writer.writerows(formatted_data)
            
        print("CSV文件已成功生成！")
    except Exception as e:
        print(f"写入CSV文件时出错: {e}")

import fire

if __name__ == '__main__':
    # INFERENCE_BASE_DIRECTORY = "open-r1/infer/sft_train_qwen2.5_usa_value_2025-07-15_16-19-28/"

    fire.Fire(aggregate_and_create_csv)