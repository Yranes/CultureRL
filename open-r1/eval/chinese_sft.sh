#!/bin/bash
#SBATCH -J cnEval
#SBATCH -o evalchina.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=50G

source ~/miniconda3/bin/activate vllm_env

conda activate vllm_env

# --- 配置区 ---
# 修改: 将国家配置更新为中国
COUNTRY_CODE="CHN"
DEMONYM="Chinese"
LANGUAGE="chinese"

# --- 核心脚本路径 (无需修改) ---
CR_SCRIPT="open-r1/infer/generate_cr_lang.py"
LOGITS_SCRIPT="open-r1/infer/generate_logits.py"
FULL_SCRIPT="open-r1/infer/generate_full.py"

# --- 数据文件路径 (自动更新) ---
WVS_DATA_PATH="open-r1/data/wvs/wvs_question2_${COUNTRY_CODE}.json"
HOFSTEDE_DATA_PATH="open-r1/data/hofstede/vsm13.json"

# --- 执行区 ---
# 将第一个命令行参数作为模型结果目录的基础路径
BASE_DIR=$1 # 示例: "open-r1/runs/Chinese/sft_train_qwen2.5_china_value_YYYY-MM-DD_HH-MM-SS/results"

if [ ! -d "$BASE_DIR" ]; then
    echo "错误: 目录不存在: $BASE_DIR"
    exit 1
fi

echo "开始处理目录: $BASE_DIR"
echo "目标国家: $COUNTRY_CODE ($DEMONYM)"
echo "---------------------------------"

# TODO: 如有需要，请在此处定义需要排除的Checkpoint
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
        echo "[1/3] 正在运行 generate_cr_lang.py..."
        # 修改: 将语言从 "arabic" 修改为使用 $LANGUAGE 变量
        python "$CR_SCRIPT" "$CKPT_PATH" "$LANGUAGE"

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