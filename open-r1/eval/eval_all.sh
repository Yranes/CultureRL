#!/bin/bash
#SBATCH -J masterEval
#SBATCH -o master_eval_%A.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 48:00:00 # 显著增加了时间，因为它要运行所有国家的评估
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=50G

# --- 使用说明 ---
# 这是一个“总调度脚本”，用于自动化对单个模型进行【全国家】的评估。
# 1. 调用各国的 `eval` 脚本，该脚本会从 /runs 读取模型并在 /infer 下生成结果。
# 2. 调用各国的 `scorer` 脚本，对 /infer 下的结果进行评分。
# 3. 将 /infer 目录下的结果文件迁移到国家专属子目录，为下一个国家做准备。
#
# 用法: sbatch this_script.sh
# 注意: 您需要在下面的配置区手动设置 TASK_NAME_FULL
# ----------------

source ~/miniconda3/bin/activate vllm_env
conda activate vllm_env

# ==============================================================================
# --- 1. 配置核心任务名称和路径 ---
# ==============================================================================

# TODO: 在此设置您要评估的任务的完整时间戳名称
TASK_NAME_FULL="full_train_qwen2.5_ALLVALUE_2025-07-21_01-47-57" # "sft_train_qwen2.5_ALLVALUE_2025-07-22_01-16-07" # open-r1/runs/ALL/full_train_qwen2.5_ALLVALUE_2025-07-21_01-47-57/results

# 输入路径: 指向包含模型 Checkpoints 的目录 (给 eval 脚本使用)
MODEL_RESULTS_DIR="open-r1/runs/ALL/${TASK_NAME_FULL}/results"
# 工作路径: 推理、评分和迁移都在这个目录下进行
INFER_WORK_PATH="open-r1/infer/${TASK_NAME_FULL}"

# 检查模型源目录是否存在
if [ ! -d "$MODEL_RESULTS_DIR" ]; then
    echo "错误: 模型源目录不存在: $MODEL_RESULTS_DIR"
    exit 1
fi

echo "模型源路径: $MODEL_RESULTS_DIR"
echo "推理工作路径: $INFER_WORK_PATH"


# ==============================================================================
# --- 2. 定义所有要评估的国家/语言列表 ---
# ==============================================================================
COUNTRIES=("arabic" "bengali" "german" "korean" "argentina" "turkish" "brazil" "chinese" "english")


# ==============================================================================
# --- 3. 主循环：按顺序调度每个国家的评估流程 ---
# ==============================================================================
for LANG in "${COUNTRIES[@]}"; do
    
    # --- 3a. 配置路径和名称 ---
    case "$LANG" in
        arabic)     DEMONYM="Iraqi" ;;
        bengali)    DEMONYM="Bangladeshi" ;;
        german)     DEMONYM="German" ;;
        korean)     DEMONYM="Korean" ;;
        argentina)    DEMONYM="Argentinian" ;;
        turkish)    DEMONYM="Turkish" ;;
        brazil) DEMONYM="Brazilian" ;;
        chinese)    DEMONYM="Chinese" ;;
        english)    DEMONYM="American" ;;
        *) echo "警告: 在配置中跳过未知语言 '$LANG'"; continue ;;
    esac

    EVAL_SCRIPT_PATH="./eval/${LANG}_sft.sh"
    SCORER_SCRIPT_PATH="./scorer/${LANG}.sh"

    echo ""
    echo "######################################################################"
    echo "### 开始调度语言: $LANG ###"
    echo "######################################################################"

    # --- 3b. 运行推理子脚本 ---
    # 此脚本从 $MODEL_RESULTS_DIR 读取 checkpoints，并默认在 $INFER_WORK_PATH 生成结果
    if [ -f "$EVAL_SCRIPT_PATH" ]; then
        echo "--- [阶段1/3] 正在执行推理脚本: $EVAL_SCRIPT_PATH ---"
        bash "$EVAL_SCRIPT_PATH" "$MODEL_RESULTS_DIR"
    else
        echo "--- [阶段1/3] 警告: 推理脚本未找到，跳过: $EVAL_SCRIPT_PATH ---"
        continue
    fi

    # --- 3c. 运行评分子脚本 ---
    # 此脚本对刚刚在 $INFER_WORK_PATH 中生成的文件进行评分
    if [ -f "$SCORER_SCRIPT_PATH" ]; then
        echo "--- [阶段2/3] 正在执行评分脚本: $SCORER_SCRIPT_PATH ---"
        bash "$SCORER_SCRIPT_PATH" "$INFER_WORK_PATH"
    else
        echo "--- [阶段2/3] 警告: 评分脚本未找到，跳过: $SCORER_SCRIPT_PATH ---"
    fi

    # --- 3d. 运行文件迁移 (功能已恢复) ---
    echo "--- [阶段3/3] 正在迁移 $LANG 的结果文件 ---"
    # 遍历 infer 目录下的所有 checkpoint
    for CKPT_PATH in $(find "$INFER_WORK_PATH" -maxdepth 1 -type d -name "checkpoint-*"); do
        DEST_DIR="${CKPT_PATH}/${DEMONYM}_eval_results"
        mkdir -p "$DEST_DIR"
        
        # 查找当前checkpoint目录下的所有【文件】并移动它们
        find "$CKPT_PATH" -maxdepth 1 -type f -exec mv -t "$DEST_DIR" {} +
        
        echo "已将 $CKPT_PATH 内的文件迁移至 $DEST_DIR"
    done
    
    # 重命名当次运行生成的 excel 文件，以国家区分
    if [ -f "$INFER_WORK_PATH/excel_scores.csv" ]; then
        mv "$INFER_WORK_PATH/excel_scores.csv" "$INFER_WORK_PATH/eval_scores_${DEMONYM}.csv"
        echo "已将 excel_scores.csv 重命名为 eval_scores_${DEMONYM}.csv"
    fi
    
    echo "### 语言 $LANG 的所有流程处理完毕 ###"

done

echo ""
echo "######################################################################"
echo "### 所有国家的评估流程均已调度完毕。脚本执行结束。 ###"
echo "######################################################################"