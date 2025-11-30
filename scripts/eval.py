'''
CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/SGPMIL/cam_uni_model_config.yaml --savedir ../folds_metrics --k_folds [0,1,2,3,4,5,6,7,8,9]
'''

# --- Standard Library ---
import pathlib
import ast
import shutil
import argparse
import sys
sys.path.append('/home/ubuntu/Projects/TransolverMIL')  # Adjust this path to your project directory i.e. parent dir of src

# --- Third Party ---
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch import seed_everything
import tqdm

# --- Project/Local Modules ---
import src
from src.lit_models import LitDetModel, LitGPModel, LitTSMIL
from custom_utils.general_calibration_error import gce
from src.main import init_loaders, determine_model_class, load_configs

# Use LaTeX for text rendering
plt.rc('text', usetex=False)
# Set Computer Modern as the font family
plt.rc('font', family='serif', serif=['cm'])
# Cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_dataset_name(config):
    """
    Extract dataset name from data_root_dir.
    Expected: /mnt/features/CAMELYON16_dataset/UNIv1_CAM16_lvl2_224_features
              -> CAMELYON16
    """
    root = pathlib.Path(config['data']['data_root_dir'])
    return root.name.split("_")[1].upper()


def extract_model_version(config):
    """
    Extract model version from checkpoint path.
    Expected: .../model_version/... -> model_version
    """
    ckpt = pathlib.Path(config['testing']['experiment_ckpt_dir'])
    version = ckpt.parents[2].name
    return version


def get_model_path(config, fold):
    ckpt_path = config['testing']['experiment_ckpt_dir']
    parts = ckpt_path.split('/')
    parts[-2] = str(fold)
    ckpt_path = pathlib.Path('/'.join(parts))
    assert ckpt_path.exists(), f'Checkpoint path {ckpt_path} does not exist'
    return ckpt_path

def evaluate(model, test_loader, config):
    model.eval()

    # Lists
    y_hat = []
    probs = []
    y = []

    # Move metrics to CPU 
    model.test_accuracy.to('cpu')
    model.test_auc.to('cpu')
    model.test_cohenkappa.to('cpu')
    model.test_confusion_matrix.to('cpu')
    model.test_ece.to('cpu')

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            X, Y = batch['img'].to(device), batch['label'].to(device)
            batch['img'] = batch['img'].to(device)
            batch['label'] = batch['label'].to(device)
            if isinstance(model, LitDetModel):
                if 'coords' in batch:
                    batch['coords'] = batch['coords'].to(device)
                model_out = model(batch)
                if config['model']['attention'] in ['dgrmil']:
                    Y_prob = F.softmax(model_out['cls'], dim=-1).cpu()
                else:
                    Y_prob = model_out['Y_prob'].cpu()
                probs.append(Y_prob)
                y.append(Y.cpu())
                y_hat.append(Y_prob.argmax(dim=-1).cpu())
            elif isinstance(model, LitGPModel):
                X = X.to(device)   
                model_out = model(X, Y)
                probs.append(model_out['y_hat'].cpu())
                y.append(Y.cpu())
                y_hat.append(model_out['y_hat'].argmax(dim=-1).cpu())
            elif isinstance(model, LitTSMIL):
                model_out = model(batch)
                Y_prob = model_out['Y_prob'].cpu()
                probs.append(Y_prob)
                y.append(Y.cpu())
                y_hat.append(Y_prob.argmax(dim=-1).cpu())
            else:
                raise ValueError(f"Model class {model.__class__} not recognized")

    
    # Concatenate
    y_hat = torch.cat(y_hat, dim=0).cpu()
    probs = torch.cat(probs, dim=0).cpu()
    y = torch.cat(y, dim=0).cpu()
    del X, Y, model_out

    # Metrics
    acc = model.test_accuracy(y_hat, y).item()
    if config['data']['num_classes'] == 2:
        auc = model.test_auc(probs[:, -1], y).item()
        kappa = model.test_cohenkappa(probs[:, -1], y).item()
        ece = model.test_ece(probs[:, -1], y).item()
        ACE = gce(labels=y.numpy(), probs=probs.numpy(), binning_scheme='adaptive', class_conditional=True, max_prob=False, norm='l1', num_bins=10)
        cm = model.test_confusion_matrix(probs[:, -1], y).detach().cpu().numpy()
    else:
        auc = model.test_auc(probs, y).item()
        kappa = model.test_cohenkappa(probs, y).item()
        cm = model.test_confusion_matrix(probs, y).detach().cpu().numpy()
        ACE = gce(labels=y.numpy(), probs=probs.numpy(), binning_scheme='adaptive', class_conditional=True, max_prob=False, norm='l1', num_bins=10)
        ece = model.test_ece(probs, y).item()

    return {'accuracy': acc, 'auc': auc, 'kappa': kappa, 'ece': ece, 'ace': ACE}, cm

def plot_cm(cm, savedir):
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.2)
    cm = cm.astype(int)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                     xticklabels=[f"Class {i}" for i in range(len(cm))],
                     yticklabels=[f"Class {i}" for i in range(len(cm))])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f"Confusion Matrix")

    plt.tight_layout()
    plt.savefig(savedir)
    plt.close()

def parse_k_folds(k_folds_str):
    try:
        k_folds = ast.literal_eval(k_folds_str)
        assert isinstance(k_folds, list) and all(isinstance(item, int) for item in k_folds)
        return k_folds
    except (ValueError, SyntaxError, AssertionError):
        raise argparse.ArgumentTypeError("k_folds must be a list of integers, e.g., [0, 1, 2, 3]")

def save_metrics(results, config, args):
    dataset_name = extract_dataset_name(config)
    feature_extractor = config['data']['feature_extractor'].upper()
    model_family = args.config.split('/')[-2]
    model_version = extract_model_version(config)

    savedir = (
        pathlib.Path(args.savedir)
        / dataset_name
        / feature_extractor
        / model_family
        / model_version
    )
    savedir.mkdir(parents=True, exist_ok=True)


    # Create DataFrame from results
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    metrics_df.to_csv(savedir / f"{model_version}_metrics_per_fold.csv", index_label="fold")

    # Calculate mean and std for each metric
    metrics_summary_stats = metrics_df.agg(['mean', 'std']).transpose()

    # Prepare summary data
    summary_data = {'num_folds': [len(metrics_df)]}  # First column with the number of folds
    
    # Populate mean and std columns
    for metric in metrics_summary_stats.index:
        summary_data[f"mean_{metric}"] = [metrics_summary_stats.at[metric, 'mean']]
        summary_data[f"std_{metric}"] = [metrics_summary_stats.at[metric, 'std']]

    # Create the summary DataFrame
    metrics_summary_df = pd.DataFrame(summary_data)

    # Save the summary DataFrame to a CSV file
    metrics_summary_df.to_csv(savedir / f"{model_version}_metrics_summary.csv", index=False)

def main(args):
    print(f'Loading config file...')
    config = load_configs(args)
    config_fpath = pathlib.Path(args.config)
    dataset_name = extract_dataset_name(config)            # CAM16 / CAMELYON16
    feature_extractor = config['data']['feature_extractor'].upper()  # UNIv1
    model_family = config_fpath.parent.name                # SGPMIL
    model_version = extract_model_version(config)          # hdims256_cluster64_mlp2

    savedir = (
        pathlib.Path(args.savedir)
        / dataset_name
        / feature_extractor
        / model_family
        / model_version
    )    
    if savedir.exists() and savedir.is_dir():
        shutil.rmtree(savedir)
    
    savedir.mkdir(parents=True, exist_ok=True)

    print(f'Setting seed {config["seed"]}...')
    seed_everything(config['seed'], workers=True)

    print(f'Loading test data...')
    _, _, test_loader = init_loaders(config)

    # Determine model class
    model_class = determine_model_class(config)
    total_cm = np.zeros((config['data']['num_classes'], config['data']['num_classes']))
    results = {}
    for fold in args.k_folds:
        print(f'Loading fold {fold}...')
        print(f'Loading model...')
        ckpt_path = get_model_path(config, fold)
        model = model_class.load_from_checkpoint(checkpoint_path=ckpt_path, 
                                                 config=config, map_location=device)

        print(f'Running evaluation...')
        results[fold], cm = evaluate(model, test_loader, config)
        del model

        total_cm += cm

        cm_savedir = pathlib.Path(savedir) / f'cm_{model_version}_fold{fold}.png'
        plot_cm(cm, cm_savedir)
    
    print('Saving results...')

    save_metrics(results, config, args)
    print('Plotting total confusion matrix...')
    cm_savedir = cm_savedir.parent / f'cm_{model_version}_total.png'
    plot_cm(total_cm, cm_savedir)

    print('Done!')


if __name__ == '__main__':
    print('Loading config file...')
    parser = argparse.ArgumentParser(description='Evaluate k-folds on test set')
    parser.add_argument('--config', "-c", type=str, default=None, help='/path/to/model/config/file')
    parser.add_argument('--exp_config', "-ec", type=str, default=None)
    parser.add_argument('--savedir', "-sd", type=str, default=None, help='/path/to/save/directory for evaluation results')
    parser.add_argument('--k_folds', type=parse_k_folds, default=[], help='List of k-folds, e.g. [0, 1, 2, 3, 4]')
    args = parser.parse_args()
    main(args)