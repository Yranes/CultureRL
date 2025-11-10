#!/bin/bash

# --- 1. 配置语言和基础路径 ---
SCORER_BASE_PATH="open-r1/infer/scorer"

LANGUAGE="german"
LANGUAGE_CAP="German"

# --- 2. 定义推理结果的基础路径 ---
INFER_BASE_PATH=$1 # "open-r1/infer/full_train_qwen2.5_german_value_2025-07-18_19-29-25" # "open-r1/infer/sft_train_qwen2.5_german_value_2025-07-17_10-41-36/" # "open-r1/infer/Qwen2.5-3B-Instruct/german" # "open-r1/infer/full_train_qwen2.5_german_value_2025-07-16_14-49-07/"

# --- 3. 检查路径是否存在 ---
if [ ! -d "$INFER_BASE_PATH" ]; then
    echo "错误: 目录不存在: $INFER_BASE_PATH"
    exit 1
fi

echo "开始处理目录: $INFER_BASE_PATH"

# --- 4. 新增逻辑：检查是否存在 checkpoint 目录 ---
CHECKPOINTS=($(find "$INFER_BASE_PATH" -maxdepth 1 -type d -name "checkpoint-*"))

# 定义一个可复用的评分函数，避免代码重复
run_scoring() {
    local TARGET_PATH=$1
    echo "--> Running test_offensEval_lang.py..."
    if [ -d "$TARGET_PATH" ]; then
        python "$SCORER_BASE_PATH/test_offensEval_lang.py" "$TARGET_PATH" "$LANGUAGE"
    else
        echo "    Warning: Input directory not found, skipping. Path: $TARGET_PATH"
    fi

    local WVS_SPA_INPUT_FILE="$TARGET_PATH/wvs_question2_DEU.json"
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


if [ ${#CHECKPOINTS[@]} -gt 0 ]; then
    # --- 模式一：检查点模式 (如果找到 checkpoint-* 目录) ---
    echo "发现 ${#CHECKPOINTS[@]} 个 checkpoint 目录，将逐个处理..."
    
    for CKPT_PATH in $(printf "%s\n" "${CHECKPOINTS[@]}" | sort -V); do
        echo ""
        echo "======================================================================"
        echo "Processing Checkpoint: $CKPT_PATH"
        echo "======================================================================"
        run_scoring "$CKPT_PATH"
    done
else
    # --- 模式二：直接扫描模式 (如果没有找到 checkpoint-* 目录) ---
    echo "未发现 checkpoint-* 目录，将直接处理基础目录: $INFER_BASE_PATH"
    echo "======================================================================"
    run_scoring "$INFER_BASE_PATH"
fi


# --- 5. 导出最终的 Excel 报告 ---
echo ""
echo "======================================================================"
echo "运行最终报告导出脚本..."
python open-r1/infer/scorer/export_excel.py "$INFER_BASE_PATH"

echo ""
echo "所有任务已处理完毕。"
echo "======================================================================"