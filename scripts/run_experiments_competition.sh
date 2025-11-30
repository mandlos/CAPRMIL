#!/bin/bash
# usage: conda activate tsmil && bash scripts/run_experiments_competition.sh  --config /path/to/config.yaml --train_frac 0.1 0.25 0.5 0.75 1. --gpu 1
# -------------------------
# 1. Baseline config
# -------------------------
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments"
BASE_CFG=""
GPU_INDEX=0

# -------------------------
# 2. Experiment grid
# -------------------------
TRAIN_FRAC=(1.0)
FOLD=(0 1 2 3 4 5 6 7 8 9)
SEED=(2025) # later for multiple seeds experiments e.g. for BRACS

# check for --hidden XX
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) BASE_CFG="$2"; shift ;;
        --seeds) IFS=',' read -r -a SEED <<< "$2"; shift ;;
        --train_frac)
            shift
            TRAIN_FRAC=()
            while [[ "$1" && "$1" != --* ]]; do
                TRAIN_FRAC+=("$1")
                shift
            done
            continue
            ;;
        --gpu) GPU_INDEX="$2"; shift ;;
    esac
    shift
done

if [[ -z "$BASE_CFG" ]]; then
  echo "ERROR: --config not provided"
  exit 1
fi


config_logging_model_ckpt_dir=$(yq -r '.logging.model_ckpt_dir' "$BASE_CFG")
DATASET_CFG=$(yq -r '.data.dataset_config' "$BASE_CFG")
data_root_basename=$(basename "$(yq -r '.data.data_root_dir' "$DATASET_CFG")")
echo "🟢 GPU Index: $GPU_INDEX"
echo "🟢 Using base config: $BASE_CFG"
echo "🟢 Model checkpoints will be saved to: $config_logging_model_ckpt_dir"
echo "🟢 Dataset root basename: $data_root_basename"
echo "dataset config: $DATASET_CFG"


# -------------------------------
# 3. Helper: get baseline values
# -------------------------------
get_baseline() {
    yq ".$1" "$BASE_CFG"
}

# -------------------------
# 4. Run all experiments
# -------------------------
for frac in "${TRAIN_FRAC[@]}"; do
for seed in "${SEED[@]}"; do
for fold in "${FOLD[@]}"; do

    # -----------------------------------------
    # build model name from ONLY changed params
    # -----------------------------------------
    NAME="${EXPERIMENT_DIR}/$(basename "$config_logging_model_ckpt_dir")"
    
    CFG_NAME="frac${frac}"
    NAME="${NAME}_${CFG_NAME}"

    NAME="${NAME}/${data_root_basename}"

    # ---------------------------------------
    # create modified config file
    # ---------------------------------------
    EXP_CFG="${NAME}/exp_${CFG_NAME}_fold${fold}.yaml"
    mkdir -p "$(dirname "$EXP_CFG")"
    echo "🟡 Creating config file: $EXP_CFG"
    cp "$BASE_CFG" "$EXP_CFG"

    # override only changed values
    yq -Yi ".logging.model_version = \"$CFG_NAME\"" "$EXP_CFG"
    yq -Yi ".data.split = $fold" "$EXP_CFG"
    yq -Yi ".training.gpu_index = [$GPU_INDEX]" "$EXP_CFG"
    yq -Yi ".seed = $seed" "$EXP_CFG"
    yq -Yi ".data.train_frac = $frac" "$EXP_CFG"



    echo "🔵 Running experiment: $NAME (fold $fold)" 

    # ---------------------------------------
    # run your model
    # ---------------------------------------
    python "$PROJECT_ROOT/src/main.py" --config "$EXP_CFG"

done
done
done