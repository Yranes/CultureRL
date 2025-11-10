#!/bin/bash
#SBATCH -J gptEval
#SBATCH -o eval_gpt_%A_%a.out  # 使用SLURM的作业ID和任务ID命名输出文件
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --mem=50G

# --- 使用说明 ---
# 这是一个用于评估 OpenAI (GPT) 模型在不同文化上表现的通用脚本。
# 通过命令行参数传入【模型名称】和【文化/语言】来执行。
#
# 用法: sbatch this_script.sh <model_name> <culture_lang>
#
# 支持的 <culture_lang> 参数:
#   arabic, bengali, german, korean, spanish, turkish, portuguese
#
# 示例:
#   sbatch this_script.sh gpt-4o german
#   sbatch this_script.sh gpt-3.5-turbo turkish
# ----------------

source ~/miniconda3/bin/activate vllm_env
conda activate vllm_env

# --- 1. 从命令行参数获取模型名称和语言 ---
MODEL_NAME=$1
CULTURE_LANG=$2

if [ -z "$MODEL_NAME" ] || [ -z "$CULTURE_LANG" ]; then
    echo "错误: 参数不足。"
    echo "用法: sbatch $0 <model_name> <culture_lang>"
    exit 1
fi

# 清理模型名称，使其适合用作目录名 (例如，移除斜杠)
MODEL_NAME_CLEAN=$(echo "$MODEL_NAME" | sed 's/\//_/g')

echo "收到的模型名称: $MODEL_NAME"
echo "收到的文化/语言参数: $CULTURE_LANG"

# --- 2. 根据语言配置国家/地区特有的变量 ---
case "$CULTURE_LANG" in
    arabic) COUNTRY_CODE="IRQ"; DEMONYM="Iraqi" ;;
    bengali) COUNTRY_CODE="BGD"; DEMONYM="Bangladeshi" ;;
    german) COUNTRY_CODE="DEU"; DEMONYM="German" ;;
    korean) COUNTRY_CODE="KOR"; DEMONYM="Korean" ;;
    spanish) COUNTRY_CODE="ARG"; DEMONYM="Argentinian" ;; # 可根据需要改为 MEX
    turkish) COUNTRY_CODE="TUR"; DEMONYM="Turkish" ;;
    portuguese) COUNTRY_CODE="BRA"; DEMONYM="Brazilian" ;;
    english) COUNTRY_CODE="USA"; DEMONYM="American" ;;
    *) echo "错误: 不支持的语言 '$CULTURE_LANG'。"; exit 1 ;;
esac

echo "已配置 -> 国家代码: $COUNTRY_CODE, 人群称谓: $DEMONYM"

# --- 3. 定义脚本和数据路径 ---
CR_SCRIPT="open-r1/infer/generate_cr_lang.py"
OPENAI_SCRIPT="open-r1/infer/generate_openai.py" # 新的OpenAI评估脚本
WVS_DATA_PATH="open-r1/data/wvs/wvs_question2_${COUNTRY_CODE}.json"
HOFSTEDE_DATA_PATH="open-r1/data/hofstede/vsm13.json"

# --- 5. 执行评估任务 ---
echo ""
echo "======================================================================"
echo "开始在 OpenAI 模型 '$MODEL_NAME' 上为文化 '$DEMONYM' 进行评估"
echo "======================================================================"

# --- 任务1: 运行 generate_cr_lang.py ---
echo "[1/3] 正在运行 generate_cr_lang.py..."
python "$CR_SCRIPT" "$MODEL_NAME" "$CULTURE_LANG"

# --- 任务2: 运行 generate_openai.py (处理 WVS 数据) ---
echo "[2/3] 正在运行 generate_openai.py (WVS)..."
# 根据您的说明，WVS数据调用时不需要 culture 参数
if [ -f "$WVS_DATA_PATH" ]; then
    python "$OPENAI_SCRIPT" "$MODEL_NAME" "$WVS_DATA_PATH" "$DEMONYM" 
else
    echo "  -> 警告: WVS数据文件未找到: $WVS_DATA_PATH"
fi

# --- 任务3: 运行 generate_openai.py (处理 VSM/Hofstede 数据) ---
echo "[3/3] 正在运行 generate_openai.py (VSM)..."
# 根据您的说明，VSM数据调用时需要 culture (即 DEMONYM) 参数
if [ -f "$HOFSTEDE_DATA_PATH" ]; then
    python "$OPENAI_SCRIPT" "$MODEL_NAME" "$HOFSTEDE_DATA_PATH" "$DEMONYM"
else
    echo "  -> 警告: Hofstede数据文件未找到: $HOFSTEDE_DATA_PATH"
fi

echo ""
echo "======================================================================"
echo "模型 '$MODEL_NAME' 在文化 '$DEMONYM' 上的评估已完成。"
echo "======================================================================"