# Example usage:
# conda activate caprmil && python main.py --c /path/to/config.yaml
# --- Standard Library ---
import os
import sys
import shutil
import pathlib
import argparse
from collections.abc import Mapping

# --- Third Party ---
import yaml
import wandb
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

# --- Project/Local Modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src
from src.lit_models import LitGPModel, LitDetModel, LitCAPRMIL
from data.dataset_generic import Generic_MIL_Dataset
from utils.utils import get_split_loader, subsample_train_split
from custom_utils.utils import EpochTimingCallback
from utils.gpu_util_callback import PeakGPUMemoryCallback
from utils.params_callback import TrainableParamsCallback

# --- Environment Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)      # Detect NaN values in gradients
torch.set_float32_matmul_precision('high')   # Use float32 for matmul


def init_loaders(config):
    # Dataset and DataLoader initialization logic
    print('\nInit loaders....', end=' ')
    dataset = Generic_MIL_Dataset(
        csv_path=config['data']['csv_path'], 
        data_dir=config['data']['data_root_dir'],
        shuffle=False, 
        seed=config['seed'], 
        print_info=config['data']['print_info'],
        label_dict=config['data']['label_dict'], 
        patient_strat=config['data']['patient_strat'],
        ignore=[], shape_file=config['data']['shape_file'],
    )
    assert isinstance(config['data']['split'], int), 'Split number must be integer'
    train_split, val_split, test_split = dataset.return_splits(
        from_id=False, 
        csv_path='{}/splits_{}.csv'.format(config['data']['split_dir'], config['data']['split'])
    )
    # Subsample train dataset if train_frac < 1.0
    if config['data']['train_frac'] < 1.0:
        train_split = subsample_train_split(train_split, train_frac=config['data']['train_frac'])

    train_loader = get_split_loader(train_split, training=True, testing=False, weighted=True, use_h5=config['data']['use_h5'])
    val_loader = get_split_loader(val_split, testing=False, use_h5=config['data']['use_h5'])
    test_loader = get_split_loader(test_split, testing=False, use_h5=config['data']['use_h5'])
    print('Done')
    return train_loader, val_loader, test_loader

def init_model(config, train_loader, val_loader):
    print('\nInit model....', end=' ')
    model = None
    if config['model']['attention'] in ['sgpmil', 'agp']:
        print('\nAttention GP model')
        model = LitGPModel(config, num_training_points=len(train_loader.dataset), 
                           num_val_points=len(val_loader.dataset))
    if config['model']['attention'] in ['clam', 'transmil', 'abmil', 'dgrmil', 'bayesmil-spvis', 'meanmil']:
        model = LitDetModel(config)
    if config['model']['attention'] == 'tsmil':
        model = LitCAPRMIL(config)

    if model is None:
        raise ValueError("Model initialization failed. Check model type in config.")
    
    print('\nDone!')
    return model
    
def init_loggers(config):
    print('\nInit loggers....', end=' ')
    logger = None
    if config['logging']['wandb']:
        wandb.login()
        wandb.finish()
        run = wandb.init(project=config['logging']['project'],
                         name=f"{config['logging']['project']}_{config['data']['feature_extractor']}_{config['model']['attention']}",
                         group=config['logging']['group'])
        logger = WandbLogger(log_model=True) 
        return logger, run
    print('Done!')
    return None, None

def init_callbacks(config):
    # Callbacks
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval=config['training']['lr_logging_interval'], 
                                        log_weight_decay=config['training']['log_weight_decay'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val/loss',
                                    patience=config['training']['patience'],
                                    verbose=True,
                                    mode='min', 
                                    min_delta=config['training']['min_delta'])
    # Epoch timing
    epoch_timing = EpochTimingCallback()

    # Model checkpoint
    if config['data']['seed_mode']:
        dpath = os.path.join(config['logging']['model_ckpt_dir']+'_'+config['logging']['model_version'],
                        config['data']['data_root_dir'].split('/')[-1],
                        str(config['seed']))
    else:
        dpath = os.path.join(config['logging']['model_ckpt_dir']+'_'+config['logging']['model_version'],
                                config['data']['data_root_dir'].split('/')[-1],
                                str(config['data']['split']))
    
    # Peak GPU utilization callback
    gpu_mem_cb = PeakGPUMemoryCallback()
    
    # Trainable params callback
    params_cb = TrainableParamsCallback()

    if config['phase']=='train':
        if os.path.exists(dpath):
            shutil.rmtree(dpath)
        os.makedirs(dpath, exist_ok=True)
    
    top_callback = ModelCheckpoint(dirpath=dpath,
                                    filename="epoch={epoch}-val_accuracy={val/balanced_accuracy:.4f}",
                                    save_top_k=3,
                                    monitor="val/balanced_accuracy",
                                    auto_insert_metric_name=False,
                                    mode="max",
                                    verbose=True)

    last_epoch_callback = ModelCheckpoint(dirpath=dpath,
                                            filename='final-epoch={epoch}',
                                            save_last=True,
                                            verbose=True)
    
    return [early_stopping, top_callback, last_epoch_callback, 
            lr_monitor, epoch_timing, gpu_mem_cb, params_cb]

def get_ckpt_path(config):
    pth = config['testing']['experiment_ckpt_dir']
    print(pth)
    config_ckpt_path = pathlib.Path(pth)
    config_ckpt_path = config_ckpt_path if not str(config_ckpt_path).endswith('/') else config_ckpt_path.parent
    # Ensure config['data']['split'] is either a str or int and that the str contains an int
    split = config['data']['split']
    if isinstance(split, str):
        assert split.isdigit(), f"config['data']['split'] should be a string containing an integer, got {split}"
        split = int(split)

    config_ckpt_path = config_ckpt_path / str(split)
    assert config_ckpt_path.is_dir(), f"{config_ckpt_path} is not a valid directory"
    
    ckpt_path = config_ckpt_path / 'last.ckpt'
    assert ckpt_path.is_file(), f"{ckpt_path} does not exist"
    
    return ckpt_path

def determine_model_class(config):
    # Returns the appropriate model class based on the configuration
    if config['model']['attention'] in ['sgpmil', 'agp']:
        model_class = LitGPModel
    elif config['model']['attention'] in ['clam', 'transmil', 'abmil', 'dgrmil', 
                                          'bayesmil', 'bayesmil-spvis', 'meanmil']:
        model_class = LitDetModel
    elif config['model']['attention'] == 'tsmil':
        model_class = LitCAPRMIL
    else:
        raise ValueError("Unknown model type in configuration")
    return model_class
    
def main(config):
    # wandb 
    logger, run = init_loggers(config)
    seed_torch(config['seed'])

    try:
        # Loaders
        train_loader, val_loader, test_loader = init_loaders(config)
        
        # Lightning model
        if config['phase'] == 'train':
            model = init_model(config, train_loader, val_loader)
        else:
            # Load Model for testing
            ckpt_path = config['testing']['experiment_ckpt_dir']
            print(f'Loading model from {ckpt_path}')
            model_class = determine_model_class(config)
            model = model_class.load_from_checkpoint(checkpoint_path=ckpt_path, config=config)

        callbacks = init_callbacks(config)

        # Trainer setup
        print('\nInit trainer....', end=' ')
        trainer = Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                          logger=logger,
                          profiler=None,
                          deterministic=True,
                          gradient_clip_val=1.0,
                          max_epochs=config['training']['max_epochs'],
                          devices=config['training']['gpu_index'],
                          strategy=config['training']['strategy'],
                          precision=config['training']['precision'],
                          callbacks=callbacks,
                          enable_progress_bar=True)
        print('Done!')

        assert config['phase'] in ['train', 'test'], 'Phase should be either train or test'
        # Training/Testing
        if config['phase']=='train':
            print('\nStart training....')
            trainer.fit(model, train_loader, val_loader)
        else:
            print('\nStart testing....')
            trainer.test(model, test_loader)
        
        print('Done!')
    finally:
        if config['logging']['wandb']:
            run.finish()
            print('Wandb run finished')

    pass

# Function from probabilistic attention GP based MIL https://github.com/arneschmidt/attention_gp/tree/main
def config_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, Mapping):
            tmp = config_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

# Function from probabilistic attention GP based MIL https://github.com/arneschmidt/attention_gp/tree/main
def load_configs(args):
    with open(args.config) as file:
        config = yaml.full_load(file)
    with open(config["data"]["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)

    config = config_update(config, config_data_dependent)

    if args.exp_config is not None:
        with open(args.exp_config) as file:
            exp_config = yaml.full_load(file)
        config = config_update(config, exp_config)

    return config

# Function from CLAM testbed
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    print('Load config file')
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--config', "-c", type=str, default='../configs/config.yaml', help='Path to the default config file')
    parser.add_argument('--exp_config', "-ec", type=str, default=None, help='Path to the experiment config file, optional, params override initial config')
    args = parser.parse_args()
    config = load_configs(args)

    if config['logging']['run_name'] == 'auto':
        config['logging']['run_name'] = args.exp_config.split('/')[-2]

    print('Create output folder')
    config['output_dir'] = os.path.join(config['data']['artifact_dir'], config['logging']['run_name'])
    os.makedirs(config['output_dir'], exist_ok=True)
    print('Output will be written to: ', config['output_dir'])       

    main(config)