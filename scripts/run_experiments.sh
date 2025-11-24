#!/bin/bash
# usage: bash scripts/run_experiments.sh
# -------------------------
# 1. Baseline config
# -------------------------
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments"
BASE_CFG="${PROJECT_ROOT}/configs/TSMIL/cam_uni_model_config.yaml"
config_logging_model_ckpt_dir=$(yq -r '.logging.model_ckpt_dir' "$BASE_CFG")
DATASET_CFG=$(yq -r '.data.dataset_config' "$BASE_CFG")
data_root_basename=$(basename "$(yq -r '.data.data_root_dir' "$DATASET_CFG")")

# -------------------------
# 2. Experiment grid
# -------------------------
HIDDEN_DIMS=(128 256 512)
CLUSTERS=(16 32 64 128)
MLP_RATIOS=(1 2 4)
FOLD=(0 1 2 3 4)
SEED=(2025) # later for multiple seeds experiments e.g. for BRACS

# -------------------------------
# 3. Helper: get baseline values
# -------------------------------
get_baseline() {
    yq ".$1" "$BASE_CFG"
}

# -------------------------
# 4. Run all experiments
# -------------------------
for hid in "${HIDDEN_DIMS[@]}"; do
for clu in "${CLUSTERS[@]}"; do
for mlp in "${MLP_RATIOS[@]}"; do
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

    NAME="${NAME}/${data_root_basename}"

    # ---------------------------------------
    # create modified config file
    # ---------------------------------------
    EXP_CFG="${NAME}/exp_${CFG_NAME}_fold${fold}.yaml"
    mkdir -p "$(dirname "$EXP_CFG")"
    echo "🟡 Creating config file: $EXP_CFG"
    cp "$BASE_CFG" "$EXP_CFG"

    # override only changed values
    yq -Yi ".model.hidden_dim = $hid" "$EXP_CFG"
    yq -Yi ".model.cluster_num = $clu" "$EXP_CFG"
    yq -Yi ".model.mlp_ratio = $mlp" "$EXP_CFG"
    yq -Yi ".logging.model_version = \"$CFG_NAME\"" "$EXP_CFG"
    yq -Yi ".data.split = $fold" "$EXP_CFG"

    echo "🔵 Running experiment: $NAME (fold $fold)" 

    # ---------------------------------------
    # run your model
    # ---------------------------------------
    python src/main.py --config "$EXP_CFG"

done
done
done
done