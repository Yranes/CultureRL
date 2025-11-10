#!/bin/bash

# --- 1. 配置语言和基础路径 ---
SCORER_BASE_PATH="open-r1/infer/scorer"

# 修改: 更新为中文配置
LANGUAGE="chinese"
LANGUAGE_CAP="Chinese"

# --- 2. 定义推理结果的基础路径 ---
# 将第一个命令行参数作为推理结果的基础路径
INFER_BASE_PATH=$1 # 示例: "open-r1/infer/gpt-4o-mini/Chinese/"

# --- 3. 检查路径是否存在 ---
if [ ! -d "$INFER_BASE_PATH" ]; then
    echo "错误: 目录不存在: $INFER_BASE_PATH"
    exit 1
fi

echo "开始处理目录: $INFER_BASE_PATH"

# --- 4. 定义可复用的评分函数 ---
run_scoring() {
    local TARGET_PATH=$1 # 将要处理的路径作为函数参数

    local OFFENS_EVAL_INPUT_DIR="$TARGET_PATH"
    echo "--> Running test_offensEval_lang.py..."
    if [ -d "$OFFENS_EVAL_INPUT_DIR" ]; then
        python "$SCORER_BASE_PATH/test_offensEval_lang.py" "$OFFENS_EVAL_INPUT_DIR" "$LANGUAGE"
    else
        echo "    Warning: Input directory not found, skipping. Path: $OFFENS_EVAL_INPUT_DIR"
    fi

    # 修改: 将国家代码从 IRQ 更新为 CHN (中国)
    local WVS_SPA_INPUT_FILE="$TARGET_PATH/wvs_question2_CHN.json"
    echo "--> Running wvs_cultureSPA.py..."
    if [ -f "$WVS_SPA_INPUT_FILE" ]; then
        python "$SCORER_BASE_PATH/wvs_cultureSPA.py" "$WVS_SPA_INPUT_FILE"
    else
        echo "    Warning: Input file not found, skipping. Path: $WVS_SPA_INPUT_FILE"
    fi

    local HOFSTEDE_INPUT_FILE="$TARGET_PATH/vsm13.json"
    echo "--> Running hofstede_test.py..."
    if [ -f "$HOFSTEDE_INPUT_FILE" ]; then
        python "$SCORER_BASE_PATH/hofstede_test.py" "$LANGUAGE_CAP" "$HOFSTEDE_INPUT_FILE"
    else
        echo "    Warning: Input file not found, skipping. Path: $HOFSTEDE_INPUT_FILE"
    fi
}


# --- 5. 检查是否存在 checkpoint 目录并选择执行模式 ---
echo "正在检查是否存在 checkpoint-* 目录..."
CHECKPOINTS=($(find "$INFER_BASE_PATH" -maxdepth 1 -type d -name "checkpoint-*"))

if [ ${#CHECKPOINTS[@]} -gt 0 ]; then
    # --- 模式一：检查点模式 (如果找到 checkpoint-* 目录) ---
    echo "发现 ${#CHECKPOINTS[@]} 个 checkpoint 目录，将逐个处理..."
    
    for CKPT_PATH in $(printf "%s\n" "${CHECKPOINTS[@]}" | sort -V); do
        echo ""
        echo "======================================================================"
        echo "Processing Checkpoint: $CKPT_PATH"
        echo "======================================================================"
        run_scoring "$CKPT_PATH" # 调用评分函数，传入checkpoint路径
    done
else
    # --- 模式二：直接扫描模式 (如果没有找到 checkpoint-* 目录) ---
    echo "未发现 checkpoint-* 目录，将直接处理基础目录: $INFER_BASE_PATH"
    echo "======================================================================"
    run_scoring "$INFER_BASE_PATH" # 调用评分函数，传入基础路径
fi

python open-r1/infer/scorer/export_excel.py "$INFER_BASE_PATH"

echo ""
echo "======================================================================"
echo "所有检查点已处理完毕。"
echo "======================================================================"