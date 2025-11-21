import numpy as np
import pdb
import os
import yaml
import pandas as pd
import pathlib
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_mil import MIL_fc, MIL_fc_mc
from src.model_clam import CLAM_SB, CLAM_MB
from src.model_transmil import TransMIL
from src.lit_models import LitGPModel, LitDetModel
from src.gp_models import *
from utils.utils import *
from utils.core_utils import Accuracy_Logger

def initiate_model(args, ckpt_path, device='cuda', config=None):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type =='transmil':
        model_dict = {'n_classes':args.n_classes}
        model = TransMIL(**model_dict)
    elif args.model_type == 'agp' and config is not None:
        print('Loading AGP model')
        with open(config) as file:
            config = yaml.full_load(file)
    
        model = LitGPModel.load_from_checkpoint(ckpt_path, config=config, 
                                                map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage).model

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    # print_network(model)

    if args.model_type == 'agp':
        _ = model.to(device)
        _ = model.eval()
        return model
    
    # print(f'Model is either CLAM, MIL or TransMIL')
    ckpt = torch.load(ckpt_path, weights_only=True)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def custom_init_model(config_dict: dict) -> nn.Module:
    if not isinstance(config_dict, dict) or not config_dict:
        raise ValueError('A non-empty config dictionary is required')
    
    ckpt_path = pathlib.Path(config_dict.get('model_arguments', {}).get('ckpt_path', ''))
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint file not found at {ckpt_path}')
    
    attention_type = config_dict.get('model_arguments', {}).get('model_type', '')
    valid_attention_types = ['agp', 'clam_sb', 'clam_mb', 'transmil']
    if attention_type not in valid_attention_types:
        raise ValueError(f'attention_type must be one of {valid_attention_types}')

    model_config_fpath = pathlib.Path(config_dict.get('model_arguments', {}).get('config', ''))
    if not model_config_fpath.exists() or not model_config_fpath.is_file():
        raise FileNotFoundError(f'Config file not found at {model_config_fpath}')
    
    with open(model_config_fpath) as file:
        model_config = yaml.full_load(file)

    print('Init Model with custom_init_model')
    if attention_type == 'agp':
        print('Loading AGP model')
        model = LitGPModel.load_from_checkpoint(
            ckpt_path, config=model_config, 
            map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage
        ).model
    else:
        print(f'Loading {attention_type.upper()} model')
        model = LitDetModel.load_from_checkpoint(
            ckpt_path, config=model_config,
            map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage
        ).model
    model.eval()
    return model

    
def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if args.model_type == 'agp':
                Y_prob, Y_prob_se, logits, attention, pre_mc_integration, _ = model(data)
                Y_hat = torch.argmax(Y_prob, dim=-1)
            else:
                logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
