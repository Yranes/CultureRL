#!/bin/bash
#SBATCH -J usaAblation                       
#SBATCH -o ablationusa.out                           
#SBATCH -p compute 
#SBATCH -N 1  
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH --mem=50G                         

export WANDB_API_KEY="your_wandb_api_key"
export WANDB_BASE_URL=https://api.bandw.top



SUFFIX=$(date +'%Y-%m-%d_%H-%M-%S')
YAML_PATH="open-r1/recipes/wvs_value/ablation/ablation_usa_full.yaml"

RUN_DIR_BASE='open-r1/runs'
RUN_NAME="ALLRule_train_qwen2.5_usa_value_${SUFFIX}"
RUN_DIR=${RUN_DIR_BASE}/ablation/${RUN_NAME}
mkdir -p "${RUN_DIR}"

LOG_PATH="${RUN_DIR}/train.log"
RESULTS_PATH="${RUN_DIR}/results/"

cp "$0" "${RUN_DIR}/run_script.sh"
cp  "$YAML_PATH" "${RUN_DIR}/"
cp open-r1/src/open_r1/rewards.py "${RUN_DIR}/"

PORT=$(shuf -i25000-30000 -n1)
NUM_PROCESS=2

. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1

CUDA_VISIBLE_DEVICES=0,1 ACCELERATE_LOG_LEVEL=info\
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes $NUM_PROCESS --main_process_port $PORT\
    src/open_r1/grpo_value.py --config $YAML_PATH\
    --output_dir $RESULTS_PATH \
    --run_name $RUN_NAME \
    --learning_rate 5e-6\
    2>&1 | tee "$LOG_PATH"