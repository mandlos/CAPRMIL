### Installation

Option 1 – using pip:
```bash
pip install -r requirements.txt
```

Option 2 – using conda:
```bash
conda env create -f environment.yaml
conda activate caprmil
```

---

## Training

To train a single model using one of the provided configuration files:

```bash
python src/main.py --config /path/to/config.yaml
```

This command launches a single training run using slide-level supervision and the settings defined in the specified YAML configuration file.

---

## Evaluation

### Slide-level evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --config /path/to/config.yaml \
    --savedir /path/to/save/metrics \
    --k_folds [0,1,2,3,4,5,6,7,8,9]
```

---

## Running Batch Experiments

The `scripts/` directory also contains shell (`.sh`) scripts for running **batch experiments**, such as cross-validation, multiple random seeds, and different training set fractions. These scripts automate large experimental sweeps and ensure reproducibility.

In contrast to single-run training, batch scripts:
- Automatically generate experiment-specific YAML configuration files
- Override parameters such as fold index, random seed, GPU index, and training fraction
- Launch multiple training runs sequentially
- Organize outputs (configs, checkpoints, logs) in a structured directory hierarchy

---

### Requirements

Before running the batch experiment scripts, ensure that:
- The CAPRMIL conda environment is activated
- `yq` is installed (used to programmatically modify YAML configuration files)
- A CUDA-enabled GPU is available

Activate the environment:
```bash
conda activate caprmil
```

If `yq` is not installed:
```bash
sudo apt-get install yq
# or
pip install yq
```

---

### Example: Running Multiple Experiments

```bash
bash scripts/run_experiments_competition.sh \
    --config /path/to/config.yaml \
    --train_frac 0.1 0.25 0.5 0.75 1.0 \
    --fold 0 1 2 3 4 \
    --gpu 0
```

This command will:
- Run experiments across multiple cross-validation folds
- Vary the fraction of training data used
- Use a fixed random seed (or multiple seeds if specified)
- Create modified configuration files for each experiment
- Launch training via `src/main.py` for each configuration

---

### Script Arguments

- `--config`  
  Path to the base YAML configuration file (required)

- `--train_frac`  
  One or more fractions of the training dataset to use (e.g. `0.1 0.5 1.0`)

- `--fold`  
  One or more cross-validation fold indices

- `--seeds`  
  One or more random seeds (optional; defaults to a single seed)

- `--gpu`  
  GPU index to use for all experiments

---

### Output Structure

Each batch experiment creates a structured output directory under:

```
experiments/
└── <model_name>_<experiment_settings>/
    └── <dataset_name>/
        ├── exp_<settings>_foldX.yaml
        ├── checkpoints/
        └── logs/
```

This layout facilitates systematic comparison across folds, seeds, and training regimes.
