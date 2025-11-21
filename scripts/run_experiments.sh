#!/bin/bash

# -------------------------
# 1. Baseline config
# -------------------------
BASE_CFG="../configs/TSMIL/baseline_tsmil.yaml"

# -------------------------
# 2. Experiment grid
# -------------------------
HIDDEN_DIMS=(128 256 512)
CLUSTERS=(16 32 64 128)
MLP_RATIOS=(1 2 4)

# -------------------------
# 3. Helper: get baseline values
# -------------------------
get_baseline() {
    yq ".$1" "$BASE_CFG"
}

BASE_HID=$(get_baseline "model.hidden_dim")
BASE_CLU=$(get_baseline "model.cluster_num")
BASE_MLP=$(get_baseline "model.mlp_ratio")

# -------------------------
# 4. Run all experiments
# -------------------------
for hid in "${HIDDEN_DIMS[@]}"; do
for clu in "${CLUSTERS[@]}"; do
for mlp in "${MLP_RATIOS[@]}"; do

    # ---------------------------------------
    # build model name from ONLY changed params
    # ---------------------------------------
    NAME="tsmil"

    [[ "$hid" != "$BASE_HID" ]] && NAME="${NAME}_h${hid}"
    [[ "$clu" != "$BASE_CLU" ]] && NAME="${NAME}_c${clu}"
    [[ "$mlp" != "$BASE_MLP" ]] && NAME="${NAME}_m${mlp}"

    # ---------------------------------------
    # create modified config file
    # ---------------------------------------
    EXP_CFG="exp_${NAME}.yaml"
    cp "$BASE_CFG" "$EXP_CFG"

    # override only changed values
    yq -i ".model.hidden_dim = $hid" "$EXP_CFG"
    yq -i ".model.cluster_num = $clu" "$EXP_CFG"
    yq -i ".model.mlp_ratio = $mlp" "$EXP_CFG"
    yq -i ".logging.model_version = \"$NAME\"" "$EXP_CFG"

    echo "🔵 Running experiment: $NAME"

    # ---------------------------------------
    # run your model
    # ---------------------------------------
    python main.py --config "$EXP_CFG"

done
done
done
