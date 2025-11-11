#!/bin/bash
#SBATCH -J koreanEval                 # 修改: 作业名称
#SBATCH -o evalkorean.out               # 修改: 输出日志文件名
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=10G


# ==============================================================================
# --- 配置区: 请根据您的实际情况修改此区域 ---
# ==============================================================================

# 1. 修改: 定义要测试的国家/文化和语言
COUNTRY_CODE="KOR"
DEMONYM="Korean"
LANG="korean"

# 2. 修改: 将具体的运行名称定义为一个变量，方便修改
# --- !! 重要：请将下面的 TASK_NAME 替换为您实际的韩国运行任务名 !! ---
# 我已将 german_value -> korean_value，您需要替换为真实的日期时间戳
TASK_NAME="full_train_qwen2.5_korean_value_2025-07-17_13-40-01"
BASE_DIR="open-r1/runs/Korean/${TASK_NAME}/results"


# 3. 定义核心脚本和数据文件路径 (这部分通常无需改动)
CR_SCRIPT="open-r1/infer/generate_cr_lang.py"
LOGITS_SCRIPT="open-r1/infer/generate_logits.py"
FULL_SCRIPT="open-r1/infer/generate_full.py"

WVS_DATA_PATH="open-r1/data/wvs/wvs_question2_${COUNTRY_CODE}.json"
HOFSTEDE_DATA_PATH="open-r1/data/hofstede/vsm13.json"

# 4. 定义需要排除的Checkpoint列表
# TODO: 请确认这个排除列表是否适用于您的新任务
EXCLUDE_CKPTS=("")
echo "排除列表: ${EXCLUDE_CKPTS[@]}"

# ==============================================================================
# --- 执行逻辑: 通常无需修改此区域 ---
# ==============================================================================

if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在: $BASE_DIR"
    echo "请务必修改脚本中的 TASK_NAME 变量为您正确的模型运行名称！"
    exit 1
fi

echo "开始处理目录: $BASE_DIR"
echo "目标国家: $COUNTRY_CODE ($DEMONYM)"
echo "---------------------------------"


for CKPT_PATH in $(find "$BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do
    if [ -d "$CKPT_PATH" ]; then
        CKPT_NAME=$(basename "$CKPT_PATH")

        if [[ " ${EXCLUDE_CKPTS[@]} " =~ " ${CKPT_NAME} " ]]; then
            echo ""
            echo "==== 跳过已处理的检查点: $CKPT_NAME ===="
            echo "---------------------------------"
            continue
        fi

        echo ""
        echo "==== 正在处理检查点: $CKPT_NAME ===="

        # --- 任务1: 运行 generate_cr_lang.py ---
        echo "[1/3] 正在运行 generate_cr_lang.py..."
        python "$CR_SCRIPT" "$CKPT_PATH" "$LANG"

        # --- 任务2: 运行 generate_logits.py ---
        echo "[2/3] 正在运行 generate_logits.py..."
        if [ -f "$WVS_DATA_PATH" ]; then
            python "$LOGITS_SCRIPT" "$CKPT_PATH" "$WVS_DATA_PATH"
        else
            echo "  -> 警告: WVS数据文件未找到: $WVS_DATA_PATH"
        fi

        # --- 任务3: 运行 generate_full.py ---
        echo "[3/3] 正在运行 generate_full.py..."
        python "$FULL_SCRIPT" "$CKPT_PATH" "$HOFSTEDE_DATA_PATH" "$DEMONYM"

        echo "==== $CKPT_NAME 处理完成 ===="
        echo "---------------------------------"
    fi
done

echo ""
echo "所有检查点均已处理完毕。"