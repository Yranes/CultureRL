#!/bin/bash
#SBATCH -J bd_sft
#SBATCH -o bangladeshsft.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --mem=50G

export WANDB_API_KEY="your_wandb_api_key"
export WANDB_BASE_URL=https://api.bandw.top

source ~/miniconda3/bin/activate openr1

conda activate openr1

SUFFIX=$(date +'%Y-%m-%d_%H-%M-%S')
# 根据您的要求，将上级目录修改为 'bangali'
YAML_PATH="open-r1/recipes/wvs_value/bengali/qwen2_5_3B_sft_bangladesh_value.yaml"

# 将语言目录从 Portuguese 修改为  (Bengali/孟加拉语)
RUN_DIR_BASE='open-r1/runs/Bengali'
RUN_NAME="sft_train_qwen2.5_bangladesh_value_${SUFFIX}"
RUN_DIR=${RUN_DIR_BASE}/${RUN_NAME}
mkdir -p "${RUN_DIR}"

LOG_PATH="${RUN_DIR}/train.log"
RESULTS_PATH="${RUN_DIR}/results/"

cp "$0" "${RUN_DIR}/run_script.sh"
cp  "$YAML_PATH" "${RUN_DIR}/"

PORT=$(shuf -i25000-30000 -n1)
NUM_PROCESS=1

. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1

accelerate launch --main_process_port $PORT --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config $YAML_PATH \
    --output_dir $RESULTS_PATH \
    --run_name $RUN_NAME \
    2>&1 | tee "$LOG_PATH"