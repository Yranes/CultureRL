#!/bin/bash
#SBATCH -J bdEval
#SBATCH -o evalbangladesh.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=50G   

source ~/miniconda3/bin/activate vllm_env

conda activate vllm_env

# 1. 配置国家/地区代码和名称
COUNTRY_CODE="BGD"
DEMONYM="Bangladeshi"

# 2. 定义三个核心脚本的路径 (无需修改)
CR_SCRIPT="open-r1/infer/generate_cr_lang.py"
LOGITS_SCRIPT="open-r1/infer/generate_logits.py"
FULL_SCRIPT="open-r1/infer/generate_full.py"

# 3. 定义数据文件路径 (无需修改，将自动使用上面的 COUNTRY_CODE)
WVS_DATA_PATH="open-r1/data/wvs/wvs_question2_${COUNTRY_CODE}.json"
HOFSTEDE_DATA_PATH="open-r1/data/hofstede/vsm13.json"

BASE_DIR="open-r1/runs/Bangali/full_train_qwen2.5_bangladesh_value_2025-07-16_10-38-58/results"

if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在: $BASE_DIR"
    echo "请务必修改脚本中的 BASE_DIR 变量为您正确的模型路径！"
    exit 1
fi

echo "开始处理目录: $BASE_DIR"
echo "目标国家: $COUNTRY_CODE ($DEMONYM)"
echo "---------------------------------"

EXCLUDE_CKPTS=("")
echo "排除列表: ${EXCLUDE_CKPTS[@]}"

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
        # 将语言从 "arabic" 修改为 "bengali"
        echo "[1/3] 正在运行 generate_cr_lang.py..."
        python "$CR_SCRIPT" "$CKPT_PATH" "bengali"

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