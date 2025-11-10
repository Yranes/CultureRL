#!/bin/bash
#SBATCH -J usaEval                       
#SBATCH -o evalusa.out                           
#SBATCH -p compute 
#SBATCH -N 1  
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=50G   

source ~/miniconda3/bin/activate vllm_env

conda activate vllm_env

COUNTRY_CODE="USA"
DEMONYM="American"
LANG="english"

# 2. 定义三个核心脚本的路径
CR_SCRIPT="open-r1/infer/generate_cr_lang.py"
LOGITS_SCRIPT="open-r1/infer/generate_logits.py"
FULL_SCRIPT="open-r1/infer/generate_full.py"

# 3. 定义数据文件路径
#    注意：这里我们用 ${COUNTRY_CODE} 使其动态化
WVS_DATA_PATH="open-r1/data/wvs/wvs_question2_${COUNTRY_CODE}.json"
HOFSTEDE_DATA_PATH="open-r1/data/hofstede/vsm13.json"

BASE_DIR="open-r1/runs/USA/full_train_qwen2.5_usa_value_2025-07-12_21-40-19/results"

if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在: $BASE_DIR"
    exit 1
fi

echo "开始处理目录: $BASE_DIR"
echo "目标国家: $COUNTRY_CODE ($DEMONYM)"
echo "---------------------------------"

EXCLUDE_CKPTS=("checkpoint-300" "checkpoint-650")
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