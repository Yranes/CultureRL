#!/bin/bash
#SBATCH -J baselineEval
#SBATCH -o eval_baseline_%A_%a.out  # 使用SLURM的作业ID和任务ID命名输出文件
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=50G

# --- 使用说明 ---
# 这是一个用于评估 Qwen baseline 模型在不同语言上表现的通用脚本。
# 通过命令行参数传入语言来执行特定语言的评估。
#
# 用法: sbatch a.sh <language>
#
# 支持的 <language> 参数:
#   arabic, bengali, german, korean, spanish, turkish
#
# 示例:
#   sbatch this_script.sh bengali
#   sbatch this_script.sh german
# ----------------

source ~/miniconda3/bin/activate vllm_env
conda activate vllm_env

# --- 1. 从命令行参数获取语言 ---
LANGUAGE=$1
if [ -z "$LANGUAGE" ]; then
    echo "错误: 未提供语言参数。"
    echo "用法: sbatch $0 <language>"
    exit 1
fi

echo "收到的语言参数: $LANGUAGE"

# --- 2. 根据语言配置国家/地区特有的变量 ---
case "$LANGUAGE" in
    arabic)
        COUNTRY_CODE="IRQ"
        DEMONYM="Iraqi"
        LANGUAGE_CAP="Arabic"
        ;;
    bengali)
        COUNTRY_CODE="BGD"
        DEMONYM="Bangladeshi"
        LANGUAGE_CAP="Bengali"
        ;;
    german)
        COUNTRY_CODE="DEU"
        DEMONYM="German"
        LANGUAGE_CAP="German"
        ;;
    korean)
        COUNTRY_CODE="KOR"
        DEMONYM="Korean"
        LANGUAGE_CAP="Korean"
        ;;
    spanish) # 假设 'spanish' 在此上下文中对应墨西哥
        COUNTRY_CODE="ARG"
        DEMONYM="Argentinian"
        LANGUAGE_CAP="Spanish"
        ;;
    turkish)
        COUNTRY_CODE="TUR"
        DEMONYM="Turkish"
        LANGUAGE_CAP="Turkish"
        ;;
    portuguese)
        COUNTRY_CODE="BRA"
        DEMONYM="Brazilian"
        LANGUAGE_CAP="Portuguese"
        ;;
    *)
        echo "错误: 不支持的语言 '$LANGUAGE'。"
        exit 1
        ;;
esac

echo "已配置 -> 语言: $LANGUAGE_CAP, 国家代码: $COUNTRY_CODE, 人群称谓: $DEMONYM"

# --- 3. 定义模型、脚本和数据路径 ---
MODEL_PATH="/home/share/models/Qwen2.5-3B-Instruct" # 直接指向 baseline 模型
CR_SCRIPT="open-r1/infer/generate_cr_lang.py"
LOGITS_SCRIPT="open-r1/infer/generate_logits.py"
FULL_SCRIPT="open-r1/infer/generate_full.py"
WVS_DATA_PATH="open-r1/data/wvs/wvs_question2_${COUNTRY_CODE}.json"
HOFSTEDE_DATA_PATH="open-r1/data/hofstede/vsm13.json"

# --- 4. 定义输出目录 ---
# 为 baseline 模型的评估结果创建一个独立的、按语言组织的目录结构
OUTPUT_DIR="open-r1/runs/baseline_eval/qwen2.5-3B-instruct/${LANGUAGE}"
mkdir -p "$OUTPUT_DIR"
echo "所有输出结果将保存到: $OUTPUT_DIR"

# --- 5. 执行评估任务 (不再需要循环) ---
# 注意：我们直接在 MODEL_PATH 上运行，而不是遍历 checkpoints
echo ""
echo "======================================================================"
echo "开始在 Baseline 模型上为语言 '$LANGUAGE' 进行评估"
echo "模型路径: $MODEL_PATH"
echo "======================================================================"

# --- !! 重要假设 !! ---
# 以下 python 命令假设您的脚本可以接受一个 --output_dir 或类似的参数来指定输出位置。
# 如果您的脚本是将结果直接输出到当前目录或模型目录，您可能需要相应调整。

# --- 任务1: 运行 generate_cr_lang.py ---
echo "[1/3] 正在运行 generate_cr_lang.py..."
# 假设脚本会将输出保存在模型路径或指定的输出路径
python "$CR_SCRIPT" "$MODEL_PATH" "$LANGUAGE" # --output_dir "$OUTPUT_DIR" # 如有需要，请取消此行注释

# --- 任务2: 运行 generate_logits.py ---
echo "[2/3] 正在运行 generate_logits.py..."
if [ -f "$WVS_DATA_PATH" ]; then
    python "$LOGITS_SCRIPT" "$MODEL_PATH" "$WVS_DATA_PATH" # --output_dir "$OUTPUT_DIR" # 如有需要，请取消此行注释
else
    echo "  -> 警告: WVS数据文件未找到: $WVS_DATA_PATH"
fi

# --- 任务3: 运行 generate_full.py ---
echo "[3/3] 正在运行 generate_full.py..."
python "$FULL_SCRIPT" "$MODEL_PATH" "$HOFSTEDE_DATA_PATH" "$DEMONYM" # --output_dir "$OUTPUT_DIR" # 如有需要，请取消此行注释

echo ""
echo "======================================================================"
echo "语言 '$LANGUAGE' 的 Baseline 评估已完成。"
echo "======================================================================"