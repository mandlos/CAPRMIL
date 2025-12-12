#!/bin/bash
# usage: conda activate tsmil && bash scripts/run_experiments.sh  --config /path/to/config.yaml --hidden 128 --cluster 16 --mlp 4 --train_frac 0.1 0.25 0.5 0.75 1. --gpu 0
# -------------------------
# 1. Baseline config
# -------------------------
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments"
BASE_CFG="${PROJECT_ROOT}/configs/TSMIL/cam_uni_model_config.yaml"

# -------------------------
# 2. Experiment grid
# -------------------------
HIDDEN_DIMS=(128)
CLUSTERS=(4)
MLP_RATIOS=(4)
FOLD=(0 1 2 3 4)
# USE_TEMPERATURE=(false)
USE_TEMPERATURE=(true)
# FOLD=(0)
SEED=(2025) # later for multiple seeds experiments e.g. for BRACS
TRAIN_FRAC=(1.0)
NUM_HEADS=(8)
AGGREGATORS=("mean") # either "mean" | "attn" | "gated_attn"
GPU_INDEX=0

# check for --hidden XX
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)
            BASE_CFG="$2"
            shift 2
            ;;
        --hidden)
            HIDDEN_DIMS=("$2")
            shift 2
            ;;
        --mlp)
            MLP_RATIOS=("$2")
            shift 2
            ;;
        --cluster)
            CLUSTERS=("$2")
            shift 2
            ;;
        --seeds)
            IFS=',' read -r -a SEED <<< "$2"
            shift 2
            ;;
        --gpu)
            GPU_INDEX="$2"
            shift 2
            ;;
        --train_frac)
            shift
            TRAIN_FRAC=()
            while [[ "$1" && "$1" != --* ]]; do
                TRAIN_FRAC+=("$1")
                shift
            done
            ;;
        --heads)
            shift
            NUM_HEADS=()
            while [[ "$1" && "$1" != --* ]]; do
                NUM_HEADS+=("$1")
                shift
            done
            ;;
        --aggregator)
            shift
            AGGREGATORS=()
            while [[ "$1" && "$1" != --* ]]; do
                AGGREGATORS+=("$1")
                shift
            done
            ;;
        --fold)
            shift
            FOLD=()
            while [[ "$1" && "$1" != --* ]]; do
                FOLD+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done


config_logging_model_ckpt_dir=$(yq -r '.logging.model_ckpt_dir' "$BASE_CFG")
DATASET_CFG=$(yq -r '.data.dataset_config' "$BASE_CFG")
data_root_basename=$(basename "$(yq -r '.data.data_root_dir' "$DATASET_CFG")")
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
for temp in "${USE_TEMPERATURE[@]}"; do
for agg in "${AGGREGATORS[@]}"; do
for frac in "${TRAIN_FRAC[@]}"; do
for hid in "${HIDDEN_DIMS[@]}"; do
for clu in "${CLUSTERS[@]}"; do
for mlp in "${MLP_RATIOS[@]}"; do
for heads in "${NUM_HEADS[@]}"; do
for seed in "${SEED[@]}"; do
for fold in "${FOLD[@]}"; do

    # -----------------------------------------
    # build model name from ONLY changed params
    # -----------------------------------------
    NAME="${EXPERIMENT_DIR}/$(basename "$config_logging_model_ckpt_dir")"

    NAME="${NAME}_hdims${hid}"
    CFG_NAME="hdims${hid}"

    NAME="${NAME}_cluster${clu}"
    CFG_NAME="${CFG_NAME}_cluster${clu}"

    NAME="${NAME}_mlp${mlp}"
    CFG_NAME="${CFG_NAME}_mlp${mlp}"

    NAME="${NAME}_heads${heads}"
    CFG_NAME="${CFG_NAME}_heads${heads}"

    NAME="${NAME}_frac${frac}"
    CFG_NAME="${CFG_NAME}_frac${frac}"

    # temperature-aware naming (only if disabled)
    if [[ "$temp" == "false" ]]; then
        NAME="${NAME}_notemp"
        CFG_NAME="${CFG_NAME}_notemp"
    fi

    # aggregation-aware naming (only if not mean)
    if [[ "$agg" != "mean" ]]; then
        NAME="${NAME}_${agg}"
        CFG_NAME="${CFG_NAME}_${agg}"
    fi


    NAME="${NAME}/${data_root_basename}"

    # ---------------------------------------
    # create modified config file
    # ---------------------------------------
    EXP_CFG="${NAME}/exp_${CFG_NAME}_fold${fold}.yaml"
    mkdir -p "$(dirname "$EXP_CFG")"
    echo "🟡 Creating config file: $EXP_CFG"
    echo "🟢 use_temperature = $temp"
    cp "$BASE_CFG" "$EXP_CFG"

    # override only changed values
    yq -Yi ".model.hidden_dim = $hid" "$EXP_CFG"
    yq -Yi ".model.cluster_num = $clu" "$EXP_CFG"
    yq -Yi ".model.mlp_ratio = $mlp" "$EXP_CFG"
    yq -Yi ".model.num_heads = $heads" "$EXP_CFG"
    yq -Yi ".logging.model_version = \"$CFG_NAME\"" "$EXP_CFG"
    yq -Yi ".data.split = $fold" "$EXP_CFG"
    yq -Yi ".training.gpu_index = [$GPU_INDEX]" "$EXP_CFG"
    yq -Yi ".seed = $seed" "$EXP_CFG"
    yq -Yi ".data.train_frac = $frac" "$EXP_CFG"
    yq -Yi ".model.use_temperature = $temp" "$EXP_CFG"
    yq -Yi ".model.aggregator = \"$agg\"" "$EXP_CFG"
    

    echo "🔵 Running experiment: $NAME (fold $fold)" 

    # ---------------------------------------
    # run your model
    # ---------------------------------------
    python "$PROJECT_ROOT/src/main.py" --config "$EXP_CFG"

done
done
done
done
done
done
done
done
done