"""
Model predictions and mask evaluation script for WSI attention-based models.

Supports:
- Inference and attention extraction
- Score mask creation (heatmaps)
- Mask comparison with ground truth
- Overlay visualization

Usage:
    python model_predictions.py --config /path/to/config.yaml --inference ...

Note:
    Be careful to correctly compute the patch size when creating the mask at a given magnification level.
    Also, pay attention to the coordinate computation. This script assumes:
    - Patching is performed at some level (e.g., level 2 for 10x, level 0 for 40x).
    - Downsample factors between levels differ by powers of 2.
    - Coordinates represent the top-left corner of the patch at level 0 (in our case corresponding to 40x magnification).
"""

# --- Standard Library ---
import os
import argparse
import pathlib
import csv
import yaml
import xml.etree.ElementTree as ET
import sys
sys.path.append('/path/to/project/dir/')  # Adjust this path to your project directory i.e. parent dir of src

# --- Third Party Libraries ---
import numpy as np
import pandas as pd
import torch
import h5py
import cv2
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

# --- Project/Local Modules ---
from src.main import init_loaders, load_configs
from custom_utils.general_calibration_error import gce
from src.model_bmil import probabilistic_MIL_Bayes_spvis as BayesMIL
from src.model_bmil_subtyping import probabilistic_MIL_Bayes_spvis as BayesMIL_subtyping
from src.model_dgrmil import DGRMIL
from src.model_transmil import TransMIL
from src.lit_models import LitGPModel, LitDetModel

# Number of Monte Carlo samples for BayesMIL inference
BAYESMIL_SAMPLES = 30

# Colormap to use for heatmap visualizations
HEATMAP_CMAP = 'Reds'

# --- XML/Annotation utilities ---
def read_xml_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []

    # Iterate over each Annotation
    for annotation in root.findall('.//Annotation'):
        points = []
        
        # Iterate over Coordinate elements
        for vertex in annotation.findall('.//Coordinate'):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            points.append((x, y))
        
        # Add the list of points as an annotation
        annotations.append(points)
    
    return annotations

def annotations_to_mask(slide_dims, annotations, downsample_factor=1):
    mask_fill = np.ones((slide_dims[1], slide_dims[0], 4), dtype=np.uint8)
    mask_out = np.ones((slide_dims[1], slide_dims[0], 4), dtype=np.uint8)
    mask_fill[:] = (255, 255, 255, 0)
    mask_out[:] = (255, 255, 255, 0)

    for point in annotations:
        poly_coords = np.array([[int(x/downsample_factor), int(y/downsample_factor)] for x, y in point], np.int32)
        if len(poly_coords)>2:
            cv2.fillPoly(mask_fill, [poly_coords], color=(255, 255, 0, 90))
            cv2.polylines(mask_out, [poly_coords], isClosed=True, color=(255, 255, 0, 255), thickness=10)

    return mask_fill, mask_out

def _init_model(args):
    # Load model config
    config_fpath = pathlib.Path(args.config)
    with open(config_fpath, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ckpt_fpath = config['testing']['experiment_ckpt_dir']
    
    print(f'Loading model from checkpoint... {ckpt_fpath}')
    if config['model']['attention'] in ['agp','sgpmil']:
        print('\nAttention GP model')
        model = LitGPModel.load_from_checkpoint(ckpt_fpath, config=config)
    elif config['model']['attention'] in ['clam', 'transmil', 'abmil', 'dgrmil', 'bayesmil-spvis']:
        print('\nAttention DET model')
        model = LitDetModel.load_from_checkpoint(ckpt_fpath, config=config)
    else:
        raise ValueError('Invalid attention model')
    print('\nDone!')
    return model, config


# --- Model Prediction / Inference ---
def _inference(model, dataloader, savedir, config, args):
    model.model.eval()
    predictions = {}
    for batch in tqdm(dataloader, total=len(dataloader)):
        if args.type == 'positive' and batch['label'].item() == 0:
            continue
        if args.type == 'negative' and batch['label'].item() == 1:
            continue
        if args.type == 'all' and batch['label'].item() not in [0, 1]:
            raise ValueError("Invalid label in the test set. Please check the dataset.")
        
        with torch.no_grad():
            if isinstance(model, LitGPModel):
                X = batch['img'].to(model.device)
                y = batch['label'].to(model.device)
                output = model.forward(X, y)
                A_samples = output['attention']
                A_mean = A_samples.mean(dim=0).squeeze(dim=0).detach().cpu().numpy()
                A_std = A_samples.std(dim=0).squeeze(dim=0).detach().cpu().numpy()
                A_std = normalize(A_std, A_std)
                A_std_inv = 1 - A_std
                A_product = A_mean * A_std_inv
                predictions['A'] = A_mean
                predictions['A_std'] = A_std
                predictions['A_std_inv'] = A_std_inv
                predictions['A_product'] = A_product
            
            elif isinstance(model, LitDetModel):
                batch['img'] = batch['img'].to(model.device)
                batch['label'] = batch['label'].to(model.device)
                
                if np.any([isinstance(model.model, i) for i in [BayesMIL, BayesMIL_subtyping]]):
                    predictions = _infer_bayesmil(model, batch)
                elif isinstance(model.model, TransMIL):
                    model.model.layer1.attn.forward = _my_forward_wrapper(model.model.layer1.attn)
                    model.model.layer2.attn.forward = _my_forward_wrapper(model.model.layer2.attn)
                    output = model.model(batch['img'])
                    x_shape = batch['img'].shape[0]
                    predictions['A'] = attention_rollout_transmil(model.model)[:x_shape].detach().cpu().numpy()
                    predictions['A'] = normalize(predictions['A'], predictions['A'])
                    # predictions['A'] = output['A'].squeeze(dim=0)[:x_shape].detach().cpu().numpy()
                elif isinstance(model.model, DGRMIL):
                    output = model.forward(batch)
                    A = output['A'].squeeze(dim=0).T
                    for i in range(A.shape[1]):
                        predictions[f'A_{i}'] = A[:, i].detach().cpu().numpy()
                else:
                    output = model.forward(batch)
                    predictions['A'] = output['A'].squeeze(dim=0).detach().cpu().numpy()

            else:
                raise ValueError("Unsupported model type")
            
        predictions['X'] = batch['coords'][:, 0]
        predictions['X'] = [int(i) for i in predictions['X']]
        predictions['Y'] = batch['coords'][:, 1]
        predictions['Y'] = [int(i) for i in predictions['Y']]

        savename = savedir / f'{batch["slide_id"][0]}'
        save_predictions(savename, predictions)

def _infer_bayesmil(model, batch):
    out_prob = 0
    out_atten = 0
    out_logits = 0
    # EXTRACT DATA UNCERTAINTY: vis_data = 0
    vis_data = 0

    Y_hats = []
    ens_prob = []
    ens_atten = []
    for _ in range(BAYESMIL_SAMPLES):
        out = model.forward(batch)
        logits, Y_prob, Y_hat, A = out['top_instance'], out['Y_prob'], out['Y_hat'], out['A']
        out_prob += Y_prob
        out_atten += A.detach()
        out_logits += logits

        Y_hats.append(Y_hat)
        ens_prob.append(torch.sum(- Y_prob * torch.log(Y_prob)).item())
        A = A.t()
        A = torch.cat([A, 1 - A], dim=1)
        ens_atten.append((- A * torch.log(A)).sum(dim=1).mean().item())
        # EXTRACT DATA UNCERTAINTY: store the vector vis_data += (- A * torch.log(A)).sum(dim = 1)
        vis_data += (- A * torch.log(A)).sum(dim=1)

    out_prob /= BAYESMIL_SAMPLES
    out_atten /= BAYESMIL_SAMPLES
    out_logits /= BAYESMIL_SAMPLES
    # EXTRACT DATA UNCERTAINTY: vis_data /= N_SAMPLES
    # vis_data size: [number of patches, 1]
    vis_data /= BAYESMIL_SAMPLES

    out_ens_prob = torch.sum(- out_prob * torch.log(out_prob)).item()
    out_atten = out_atten.t()
    out_atten_save = out_atten.flatten().cpu().clone().detach().numpy()
    out_atten = torch.cat([out_atten, 1 - out_atten], dim=1)
    out_ens_atten = (- out_atten * torch.log(out_atten)).sum(dim=1).mean().item()
    # EXTRACT TOTAL UNCERTAINTY: vis_total = (- out_atten * torch.log(out_atten)).sum(dim = 1)
    # vis_total size: [number of patches, 1]
    vis_total = (- out_atten * torch.log(out_atten)).sum(dim=1)

    # EXTRACT MODEL UNCERTAINTY: vis_total - vis_data
    vis_model = vis_total - vis_data

    ens_prob = np.mean(ens_prob)
    ens_atten = np.mean(ens_atten)
    Y_hat = torch.mode(torch.cat(Y_hats, dim=1))[0]
    Y_hat = Y_hat.cpu().numpy()[0]

    return {'A': out_atten_save, 'Data_unc': vis_data.cpu().numpy(),
            'Model_unc': vis_model.cpu().numpy(), 'Total_unc': vis_total.cpu().numpy()}

def _my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.to_qkv(x).reshape(B, N, 3, attn_obj.heads, C // attn_obj.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   
        # print(q.shape, k.shape, v.shape)
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        # attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.to_out(x)
        # x = attn_obj.proj_drop(x)
        return x, 0
    return my_forward

def attention_rollout_transmil(model):
    '''
    input: list of self attention matrices
    output: attention rollout matrix
    '''
    A0 = model.layer1.attn.attn_map.mean(dim=1).squeeze(0).detach()
    A1 = model.layer2.attn.attn_map.mean(dim=1).squeeze(0).detach()
    # print(A0.shape, A1.shape)

    attn_matrices = [A0, A1]
    attn_rollout = []
    I = torch.eye(attn_matrices[0].shape[-1]).to(attn_matrices[0].device)
    prod = I
    for i, attn_matrix in enumerate(attn_matrices):
        prod = prod @ (attn_matrix + I)
        prod = prod / torch.sum(prod, dim=-1, keepdim=True)
        attn_rollout.append(prod)
    a1_cls = attn_rollout[1][0, 2:]
    return a1_cls



# --- Scoring and Mask Generation ---
def save_predictions(savepath, predictions):
    os.makedirs(savepath.parent, exist_ok=True)
    
    # Save as CSV
    with open(f"{savepath}.csv", mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = list(predictions.keys())
        writer.writerow(header)
        for i in range(len(next(iter(predictions.values())))):
            row = [predictions[k][i] for k in header]
            writer.writerow(row)
    
    # Save as H5
    with h5py.File(f"{savepath}.h5", "w") as h5_file:
        for k, v in predictions.items():
            h5_file.create_dataset(k, data=v)

def normalize(data, total_data, eps=1e-8):
    mx = max(total_data)
    mn = min(total_data)
    denom = mx - mn
    if denom < eps:
        denom = eps
    return np.array([(float(i) - mn) / denom for i in data])

def read_shapes(shape_file):
    shape_dict = {}
    with open(shape_file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line_records = line.split(',')
            shape_dict[line_records[0].split('.')[0]] = [int(line_records[1]), int(line_records[2])]
    return shape_dict

def _read_scores(fdir):
    fpaths = list(fdir.glob('*.h5'))

    scores = {}
    for fpath in fpaths:
        scores[fpath.stem] = {}
        with h5py.File(fpath, 'r') as f:
            for key in f.keys():
                scores[fpath.stem][key] = f[key][:]
    return scores

def _score_mask(slide_id, scores, shapes, args, model_name, patch_size, savedir, bin_thresholds):
    # Slide shape
    slide_shape = (shapes[0] // (2**args.level), shapes[1] // (2**args.level))
    assert model_name in ['ABMIL', 'CLAM', 'TRANSMIL', 'DGRMIL', 'AGP', 'BAYESMIL-SPVIS', 'SGPMIL']
    if model_name in ['ABMIL', 'CLAM', 'TRANSMIL']:
        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        A = normalize(scores['A'], scores['A'])
        for score, x, y in tqdm(zip(A, scores['X'], scores['Y']), desc='Creating attention masks for ABMIL/CLAM/TransMIL', total=len(A), position=1, leave=False):
            x = np.ceil(x // (2**args.level)).astype(int)
            y = np.ceil(y // (2**args.level)).astype(int)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_binary_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

    elif model_name in ['AGP', 'SGPMIL']:
        A = normalize(scores['A'], scores['A'])
        A_std = normalize(scores['A_std'], scores['A_std'])
        A_std_inv = normalize(scores['A_std_inv'], scores['A_std_inv'])
        A_product = normalize(scores['A_product'], scores['A_product'])

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A, scores['X'], scores['Y']), desc='Creating attention masks for AGP/SGPMIL', total=len(A), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_binary_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A_std, scores['X'], scores['Y']), desc='Creating attention_std masks for AGP/SGPMIL', total=len(A_std), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_std'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)
        
        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A_std_inv, scores['X'], scores['Y']), desc='Creating attention_std_inv masks for AGP/SGPMIL', total=len(A_std_inv), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_std_inv'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)
        
        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A_product, scores['X'], scores['Y']), desc='Creating attention_product masks for AGP/SGPMIL', total=len(A_product), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_product'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_product_binary_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

    elif model_name in ['BAYESMIL-SPVIS']:
        A = normalize(scores['A'], scores['A'])
        Data_unc = normalize(scores['Data_unc'], scores['Data_unc'])
        Model_unc = normalize(scores['Model_unc'], scores['Model_unc'])
        Total_unc = normalize(scores['Total_unc'], scores['Total_unc'])

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A, scores['X'], scores['Y']), desc='Creating attention masks for BAYESMIL-SPVIS', total=len(A), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_binary_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(Data_unc, scores['X'], scores['Y']), desc='Creating data_unc masks for BAYESMIL-SPVIS', total=len(Data_unc), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Data_unc'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)
    
        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Data_unc_binary_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(Model_unc, scores['X'], scores['Y']), desc='Creating model_unc masks for BAYESMIL-SPVIS', total=len(Model_unc), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Model_unc'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Model_unc_binary_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(Total_unc, scores['X'], scores['Y']), desc='Creating total_unc masks for BAYESMIL-SPVIS', total=len(Total_unc), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Total_unc'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

    elif model_name in ['DGRMIL']:
        A0 = normalize(scores['A_0'], scores['A_0'])
        A1 = normalize(scores['A_1'], scores['A_1'])
        A2 = normalize(scores['A_2'], scores['A_2'])
        A3 = normalize(scores['A_3'], scores['A_3'])
        A4 = normalize(scores['A_4'], scores['A_4'])
        A5 = normalize(scores['A_5'], scores['A_5'])

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A0, scores['X'], scores['Y']), desc='Creating attention_0 masks for DGRMIL', total=len(A0), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_0'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_binary_0_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A1, scores['X'], scores['Y']), desc='Creating attention_1 masks for DGRMIL', total=len(A1), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_1'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_binary_1_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A2, scores['X'], scores['Y']), desc='Creating attention_2 masks for DGRMIL', total=len(A2), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_2'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        for thresh in bin_thresholds:
            _, binary_mask = cv2.threshold(img, thresh * 255, 255, cv2.THRESH_BINARY)
            savepath = savepath.parent / f'Attention_binary_2_{thresh}'
            savepath.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(savepath / f'{slide_id}.jpg', binary_mask)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A3, scores['X'], scores['Y']), desc='Creating attention_3 masks for DGRMIL', total=len(A3), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_3'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A4, scores['X'], scores['Y']), desc='Creating attention_4 masks for DGRMIL', total=len(A4), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_4'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)

        img = np.zeros((slide_shape[1], slide_shape[0], 1), dtype=np.uint8)
        for score, x, y in tqdm(zip(A5, scores['X'], scores['Y']), desc='Creating attention_5 masks for DGRMIL', total=len(A5), position=1, leave=False):
            x = x // (2**args.level)
            y = y // (2**args.level)
            img[y:y+patch_size, x:x+patch_size, :] = score * 255
        savepath = savedir / 'Attention_5'
        savepath.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(savepath / f'{slide_id}.jpg', img)
    else:
        raise ValueError("Unsupported model type")

def score_masks(scores, args, savedir, bin_thresholds):
    scoremasks_savedir = savedir / 'score_masks'
    scoremasks_savedir.mkdir(parents=True, exist_ok=True)

    patch_size = np.ceil(args.patch_size // (2 ** (args.level - args.patching_level))).astype(int)

    shape_file = pathlib.Path(args.shape_file)
    shapes = read_shapes(shape_file)

    model_name = savedir.parent.stem
    for slide in tqdm(scores.keys(), desc='Creating score masks', position=0, total=len(scores)):
        _score_mask(slide, scores[slide], shapes[slide], args, model_name, patch_size, scoremasks_savedir, bin_thresholds)



# --- Metrics & Evaluation ---
def iou_coverage_etc(truth, prediction_bin, prediction_raw, slide_id, args):
    if args.type in ['negative', 'all']:
        iou, gt_coverage, auc, froc, sens, precision, recall, f1 = -1, -1, -1, -1, -1, -1, -1, -1
        annotations = truth.reshape((1, -1)).flatten()
        predictions = prediction_raw.reshape((1, -1)).flatten()

        acc = metrics.balanced_accuracy_score(annotations, predictions)
    else:
        union = truth + prediction_raw
        union_nums = np.sum(union > 0)
        intersection_nums = np.sum(union==2)
        iou = intersection_nums / union_nums
        gt_coverage = intersection_nums / np.sum(truth == 1)

        annotations = truth.reshape((1, -1)).flatten()
        predictions_raw = prediction_raw.reshape((1, -1)).flatten()
        predictions_bin = prediction_bin.reshape((1, -1)).flatten()
        
        acc = metrics.balanced_accuracy_score(annotations, predictions_bin)
        precision = metrics.precision_score(annotations, predictions_bin, zero_division=0)
        recall = metrics.recall_score(annotations, predictions_bin, zero_division=0)
        f1 = metrics.f1_score(annotations, predictions_bin, zero_division=0)
        auc = metrics.roc_auc_score(annotations, predictions_raw)
        froc, sens = compute_froc(annotations.copy().astype(np.float64), predictions_bin.copy().astype(np.float64))
        ACE = gce(labels=annotations, probs=predictions_raw, binning_scheme='adaptive', class_conditional=True, max_prob=False, norm='l1', num_bins=10)

    return {'slide_id': slide_id, 'IOU': iou, 'COVERAGE': gt_coverage, 'ACC': acc, 'PRECISION': precision, 
            'RECALL': recall, 'F1': f1, 'FROC': froc, 'AUC': auc, 'avg_sensitivity': sens, 'ACE': ACE}

def compute_froc(truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(truth, prediction, pos_label=1)
    fps = fpr * (truth.shape[0] - np.sum(truth)) / truth.shape[0]
    froc = metrics.auc(fps, tpr)

    # Compute the average sensitivity
    thrshlds = [1/4, 1/2, 1, 2, 4, 8]
    sens = []
    for thr in thrshlds:
        for i in range(len(fps) - 1):
            if fps[i] <= thr and fps[i + 1] > thr:
                sens.append((tpr[i] + tpr[i + 1]) / 2)
    sens.append(tpr[-1])
    avg_sens = np.mean(sens)

    return froc, avg_sens

def _compare_masks(score_masks_fpaths_bin, score_masks_fpaths_raw, mask_fdir, lvl, args):
        all_results = []
        for fp_bin, fp_raw in tqdm(zip(score_masks_fpaths_bin, score_masks_fpaths_raw), desc='Computing metrics', total=len(score_masks_fpaths_bin), position=1, leave=False):
            slide_id = fp_bin.stem
            score_mask_bin = cv2.imread(str(fp_bin), cv2.IMREAD_GRAYSCALE).astype(np.int32) // 255
            score_mask_raw = cv2.imread(str(fp_raw), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
            
            if args.type == 'positive':
                truth_mask = mask_fdir / f'{slide_id}_mask.tif'
                assert truth_mask.exists(), f"Truth mask {truth_mask} does not exist."

                truth_mask = openslide.OpenSlide(truth_mask)
                truth_mask = np.array(truth_mask.read_region((0, 0), lvl, truth_mask.level_dimensions[lvl]))[:, :, 0] // 255
            elif args.type == 'negative':
                # Create a numpy array with the same shape and type as the score mask filled with 0
                truth_mask = np.zeros_like(score_mask_bin)
            elif args.type == 'all':
                # if the truth mask does not exist, create a numpy array with the same shape and type as the score mask filled with 0
                truth_mask = mask_fdir / f'{slide_id}_mask.tif'
                if not truth_mask.exists():
                    truth_mask = np.zeros_like(score_mask_bin)
                else:
                    truth_mask = openslide.OpenSlide(truth_mask)
                    truth_mask = np.array(truth_mask.read_region((0, 0), lvl, truth_mask.level_dimensions[lvl]))[:, :, 0] // 255
            else:
                raise ValueError("Invalid type. Please choose 'positive', 'negative', or 'all'.")

            assert truth_mask.shape == score_mask_bin.shape, f"Truth mask {truth_mask} and score mask {score_mask_bin} shapes do not match for slide {slide_id}."

            # Compute metrics
            results = iou_coverage_etc(truth_mask, score_mask_bin, score_mask_raw, slide_id, args)
            all_results.append(results)
        
        return all_results

def _compute_metrics(mask_fdir, savedir, bin_thresholds, args):
    model_type = savedir.parent.parent.stem
    lvl = args.level
    
    for thresh in tqdm(bin_thresholds, desc='Computing metrics', position=0, total=len(bin_thresholds)):
        if model_type in ['ABMIL', 'CLAM', 'TRANSMIL']:
            score_masks_fdir_bin = savedir.parent / 'score_masks' / f'Attention_binary_{thresh}'
            score_masks_fdir_raw = savedir.parent / 'score_masks' / 'Attention'
            assert score_masks_fdir_bin.exists(), f"Score masks directory {score_masks_fdir_bin} does not exist."
            assert score_masks_fdir_raw.exists(), f"Score masks directory {score_masks_fdir_raw} does not exist."
            score_masks_fpaths_bin = list(score_masks_fdir_bin.glob('*.jpg'))
            score_masks_fpaths_raw = list(score_masks_fdir_raw.glob('*.jpg'))

            all_results = _compare_masks(score_masks_fpaths_bin, score_masks_fpaths_raw, mask_fdir, lvl, args)

            df = pd.DataFrame.from_dict(all_results)
            savepath =  savedir / f'Attention_all_{thresh}.csv'
            df.to_csv(savepath, index=False)

            means = df.mean(axis=0, numeric_only=True)
            means.to_frame(name='mean').T.to_csv(savepath.parent / f'Attention_mean_{thresh}.csv', index=False)

        elif model_type in ['AGP', 'SGPMIL']:
            score_masks_attn_bin = savedir.parent / 'score_masks' / f'Attention_binary_{thresh}'
            score_masks_attn_raw = savedir.parent / 'score_masks' / 'Attention'
            # score_masks_product = savedir.parent / 'score_masks' / f'Attention_product_binary_{thresh}'
            assert score_masks_attn_bin.exists(), f"Score masks directory {score_masks_attn_bin} does not exist."
            assert score_masks_attn_raw.exists(), f"Score masks directory {score_masks_attn_raw} does not exist."
            # assert score_masks_product.exists(), f"Score masks directory {score_masks_product} does not exist."

            score_masks_attn_fpaths_bin = list(score_masks_attn_bin.glob('*.jpg'))
            score_masks_attn_fpaths_raw = list(score_masks_attn_raw.glob('*.jpg'))
            # score_masks_product_fpaths = list(score_masks_product.glob('*.jpg'))

            all_results_attn = _compare_masks(score_masks_attn_fpaths_bin, score_masks_attn_fpaths_raw, mask_fdir, lvl, args)
            # all_results_product = _compare_masks(score_masks_product_fpaths, mask_fdir, lvl, args)
            df_attn = pd.DataFrame.from_dict(all_results_attn)
            # df_product = pd.DataFrame.from_dict(all_results_product)

            savepath_attn =  savedir / f'Attention_all_{thresh}.csv'
            # savepath_product =  savedir / f'Attention_product_all_{thresh}.csv'
            df_attn.to_csv(savepath_attn, index=False)
            # df_product.to_csv(savepath_product, index=False)

            means_attn = df_attn.mean(axis=0, numeric_only=True)
            means_attn.to_frame(name='mean').T.to_csv(savepath_attn.parent / f'Attention_mean_{thresh}.csv', index=False)
            # means_product = df_product.mean(axis=0, numeric_only=True)
            # means_product.to_frame(name='mean').T.to_csv(savepath_product.parent / f'Attention_product_mean_{thresh}.csv', index=False)

        elif model_type in ['BAYESMIL-SPVIS']:
            score_masks_attn_bin = savedir.parent / 'score_masks' / f'Attention_binary_{thresh}'
            score_masks_attn_raw = savedir.parent / 'score_masks' / 'Attention'
            # score_masks_data_unc = savedir.parent / 'score_masks' / f'Data_unc_binary_{thresh}'
            # score_masks_model_unc = savedir.parent / 'score_masks' / f'Model_unc_binary_{thresh}'
            assert score_masks_attn_bin.exists(), f"Score masks directory {score_masks_attn_bin} does not exist."
            assert score_masks_attn_raw.exists(), f"Score masks directory {score_masks_attn_raw} does not exist."
            # assert score_masks_data_unc.exists(), f"Score masks directory {score_masks_data_unc} does not exist."
            # assert score_masks_model_unc.exists(), f"Score masks directory {score_masks_model_unc} does not exist."

            score_masks_attn_fpaths_bin = list(score_masks_attn_bin.glob('*.jpg'))
            score_masks_attn_fpaths_raw = list(score_masks_attn_raw.glob('*.jpg'))
            # score_masks_data_unc_fpaths = list(score_masks_data_unc.glob('*.jpg'))
            # score_masks_model_unc_fpaths = list(score_masks_model_unc.glob('*.jpg'))

            all_results_attn = _compare_masks(score_masks_attn_fpaths_bin, score_masks_attn_fpaths_raw, mask_fdir, lvl, args)
            # all_results_data_unc = _compare_masks(score_masks_data_unc_fpaths, mask_fdir, lvl, args)
            # all_results_model_unc = _compare_masks(score_masks_model_unc_fpaths, mask_fdir, lvl, args)
            df_attn = pd.DataFrame.from_dict(all_results_attn)
            # df_data_unc = pd.DataFrame.from_dict(all_results_data_unc)
            # df_model_unc = pd.DataFrame.from_dict(all_results_model_unc)

            savepath_attn =  savedir / f'Attention_all_{thresh}.csv'
            # savepath_data_unc =  savedir / f'Data_unc_all_{thresh}.csv'
            # savepath_model_unc =  savedir / f'Model_unc_all_{thresh}.csv'

            df_attn.to_csv(savepath_attn, index=False)
            # df_data_unc.to_csv(savepath_data_unc, index=False)
            # df_model_unc.to_csv(savepath_model_unc, index=False)

            means_attn = df_attn.mean(axis=0, numeric_only=True)
            means_attn.to_frame(name='mean').T.to_csv(savepath_attn.parent / f'Attention_mean_{thresh}.csv', index=False)
            # means_data_unc = df_data_unc.mean(axis=0, numeric_only=True)
            # means_data_unc.to_frame(name='mean').T.to_csv(savepath_data_unc.parent / f'Data_unc_mean_{thresh}.csv', index=False)
            # means_model_unc = df_model_unc.mean(axis=0, numeric_only=True)
            # means_model_unc.to_frame(name='mean').T.to_csv(savepath_model_unc.parent / f'Model_unc_mean_{thresh}.csv', index=False)
        elif model_type in ['DGRMIL']:
            score_masks_attn_0_bin = savedir.parent / 'score_masks' / f'Attention_binary_0_{thresh}'
            score_masks_attn_0_raw = savedir.parent / 'score_masks' / 'Attention_0'
            score_masks_attn_1_bin = savedir.parent / 'score_masks' / f'Attention_binary_1_{thresh}'
            score_masks_attn_1_raw = savedir.parent / 'score_masks' / 'Attention_1'
            # score_masks_attn_2 = savedir.parent / 'score_masks' / f'Attention_binary_2_{thresh}'

            assert score_masks_attn_0_bin.exists(), f"Score masks directory {score_masks_attn_0_bin} does not exist."
            assert score_masks_attn_0_raw.exists(), f"Score masks directory {score_masks_attn_0_raw} does not exist."
            assert score_masks_attn_1_bin.exists(), f"Score masks directory {score_masks_attn_1_bin} does not exist."
            assert score_masks_attn_1_raw.exists(), f"Score masks directory {score_masks_attn_1_raw} does not exist."
            # assert score_masks_attn_2.exists(), f"Score masks directory {score_masks_attn_2} does not exist."

            score_masks_fpaths_0_bin = list(score_masks_attn_0_bin.glob('*.jpg'))
            score_masks_fpaths_0_raw = list(score_masks_attn_0_raw.glob('*.jpg'))
            score_masks_fpaths_1_bin = list(score_masks_attn_1_bin.glob('*.jpg'))
            score_masks_fpaths_1_raw = list(score_masks_attn_1_raw.glob('*.jpg'))

            all_results_0 = _compare_masks(score_masks_fpaths_0_bin, score_masks_fpaths_0_raw, mask_fdir, lvl, args)
            all_results_1 = _compare_masks(score_masks_fpaths_1_bin, score_masks_fpaths_1_raw, mask_fdir, lvl, args)
            # all_results_2 = _compare_masks(score_masks_fpaths[2], mask_fdir, lvl, args)

            df_0 = pd.DataFrame.from_dict(all_results_0)
            savepath =  savedir / f'Attention_all_0_{thresh}.csv'
            df_0.to_csv(savepath, index=False)
            df_1 = pd.DataFrame.from_dict(all_results_1)
            savepath =  savedir / f'Attention_all_1_{thresh}.csv'
            df_1.to_csv(savepath, index=False)
            # df_2 = pd.DataFrame.from_dict(all_results_2)
            # savepath =  savedir / f'Attention_all_2_{thresh}.csv'
            # df_2.to_csv(savepath, index=False)

            means_0 = df_0.mean(axis=0, numeric_only=True)
            means_0.to_frame(name='mean').T.to_csv(savepath.parent / f'Attention_mean_0_{thresh}.csv', index=False)
            means_1 = df_1.mean(axis=0, numeric_only=True)
            means_1.to_frame(name='mean').T.to_csv(savepath.parent / f'Attention_mean_1_{thresh}.csv', index=False)
            # means_2 = df_2.mean(axis=0, numeric_only=True)
            # means_2.to_frame(name='mean').T.to_csv(savepath.parent / f'Attention_mean_2_{thresh}.csv', index=False)
        else:
            raise ValueError("Unsupported model type")


# --- Visualization ---
def _overlay_masks(loader, args, savedir):
    cmap = plt.get_cmap(HEATMAP_CMAP)
    mask_dir = savedir.parent / 'score_masks'
    assert mask_dir.exists(), f"Mask directory {mask_dir} does not exist."

    slide_dir = pathlib.Path(args.slide_dir) 
    assert slide_dir.exists(), f"Slide directory {slide_dir} does not exist."

    for batch in tqdm(loader, total=len(loader), desc='Overlaying masks', position=0, leave=False):
        if batch['label'] == 0: continue

        slide_id = batch['slide_id'][0]
        slide_fpath = slide_dir / f'{slide_id}.tif'
        assert slide_fpath.exists(), f"Slide {slide_fpath} does not exist."
        
        slide = openslide.OpenSlide(slide_fpath)
        dims = slide.level_dimensions[args.level]
        slide_img = cv2.cvtColor(np.array(slide.read_region((0, 0), args.level, dims)), cv2.COLOR_RGBA2RGB)

        model_name = savedir.parent.parent.stem
        if model_name in ['DGRMIL']:
            # Read heatmaps
            raw_score_fpath = mask_dir / f'Attention_0/{slide_id}.jpg'
            bin_score_fpath = mask_dir / f'Attention_binary_0_0.1/{slide_id}.jpg'
            raw_scoremask = cv2.imread(str(raw_score_fpath), cv2.IMREAD_GRAYSCALE)
            bin_scoremask = cv2.imread(str(bin_score_fpath), cv2.IMREAD_GRAYSCALE)
            raw_scoremap, bin_scoremap = cmap(raw_scoremask), cmap(bin_scoremask)
            raw_min, raw_max, bin_min, bin_max = raw_scoremap.min(), raw_scoremap.max(), bin_scoremap.min(), bin_scoremap.max()
            raw_scoremap = (raw_scoremap[:, :, :3] * 255).astype(np.uint8)
            bin_scoremap = (bin_scoremap[:, :, :3] * 255).astype(np.uint8)
            
            # Read & plot annotations
            annotations_fpath = pathlib.Path(args.annotations_dir) / f'{slide_id}.xml'
            annotations = read_xml_annotations(annotations_fpath)
            maskfill, maskout = annotations_to_mask(dims, annotations, slide.level_downsamples[args.level])
            maskfill = cv2.cvtColor(maskfill, cv2.COLOR_RGBA2RGB)
            
            # Overlays between scoremaps and annotations/raw_maskfill i.e. the surface of the ground-truth annotations' polygon
            raw_overlay = cv2.addWeighted(raw_scoremap, 0.8, maskfill, 0.2, 0)
            bin_overlay = cv2.addWeighted(bin_scoremap, 0.8, maskfill, 0.2, 0)
            
        elif model_name in ['ABMIL', 'CLAM', 'TRANSMIL', 'AGP', 'SGPMIL', 'BAYESMIL-SPVIS']:
            # Read heatmaps
            raw_score_fpath = mask_dir / f'Attention/{slide_id}.jpg'
            bin_score_fpath = mask_dir / f'Attention_binary_0.1/{slide_id}.jpg'
            raw_scoremask = cv2.imread(str(raw_score_fpath), cv2.IMREAD_GRAYSCALE)
            bin_scoremask = cv2.imread(str(bin_score_fpath), cv2.IMREAD_GRAYSCALE)
            raw_scoremap, bin_scoremap = cmap(raw_scoremask), cmap(bin_scoremask)
            raw_min, raw_max, bin_min, bin_max = raw_scoremap.min(), raw_scoremap.max(), bin_scoremap.min(), bin_scoremap.max()
            raw_scoremap = (raw_scoremap[:, :, :3] * 255).astype(np.uint8)
            bin_scoremap = (bin_scoremap[:, :, :3] * 255).astype(np.uint8)
            
            # Read & plot annotations
            annotations_fpath = pathlib.Path(args.annotations_dir) / f'{slide_id}.xml'
            annotations = read_xml_annotations(annotations_fpath)
            maskfill, maskout = annotations_to_mask(dims, annotations, slide.level_downsamples[args.level])
            maskfill = cv2.cvtColor(maskfill, cv2.COLOR_RGBA2RGB)
            
            # Overlays between scoremaps and annotations/raw_maskfill i.e. the surface of the ground-truth annotations' polygon
            raw_overlay = cv2.addWeighted(raw_scoremap, 0.8, maskfill, 0.2, 0)
            bin_overlay = cv2.addWeighted(bin_scoremap, 0.8, maskfill, 0.2, 0)
        else:
            raise ValueError("Unsupported model type")

        # Plot the final overlays--Raw
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(slide_img, alpha=1)
        ax.imshow(raw_overlay, alpha=.7)
        ax.imshow(maskout, alpha=1)
        ax.axis('off')
        norm = plt.Normalize(vmin=raw_min, vmax=raw_max)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), ax=ax, orientation='vertical', shrink=0.65, label='Attention')
        sdir = savedir / raw_score_fpath.parent.stem
        sdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(sdir / f'{slide_id}.jpg', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Plot the final overlays--Binary
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        ax1.imshow(slide_img, alpha=1)
        ax1.imshow(bin_overlay, alpha=.7)
        ax1.imshow(maskout, alpha=1)
        ax1.axis('off')
        norm = plt.Normalize(vmin=bin_min, vmax=bin_max)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), ax=ax1, orientation='vertical', shrink=0.65, label='Attention binarized @ 0.1')
        sdir = savedir / bin_score_fpath.parent.stem
        sdir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(sdir / f'{slide_id}.jpg', dpi=300, bbox_inches='tight')
        plt.close(fig1)




def main(args):
    # Dataloader & model
    model, _ = _init_model(args)
    config = load_configs(args)
    _, _, test_loader = init_loaders(config)

    # Inference & save predictions
    savedir = pathlib.Path(args.savedir)
    inf_savedir = savedir / args.type / config['model']['attention'].upper() / config['logging']['model_version'] / 'scores'
    if args.inference:
        inf_savedir.mkdir(parents=True, exist_ok=True)
        _inference(model, test_loader, inf_savedir, config, args)

    # Read predictions & create masks
    bin_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    if args.compute_masks:
        scores_fdir = inf_savedir.parent
        scores = _read_scores(inf_savedir)
        score_masks(scores, args, scores_fdir, bin_thresholds)

    # Read binary masks
    if args.compare_masks:
        binary_mask_dir = pathlib.Path(args.mask_dir)
        results_savedir = inf_savedir.parent / 'results'
        results_savedir.mkdir(parents=True, exist_ok=True)
        _compute_metrics(binary_mask_dir, results_savedir, bin_thresholds, args)

    if args.overlay_masks:
        overlay_savedir = inf_savedir.parent / 'overlaid_masks'
        overlay_savedir.mkdir(parents=True, exist_ok=True)
        _overlay_masks(test_loader, args, overlay_savedir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model predictions.")
    parser.add_argument('--config', type=str, default='/path/to/config.yaml', help='Path to the config file.')
    parser.add_argument('--savedir', type=str, default='/path/to/save/results', help='Directory to save the model predictions.')
    parser.add_argument('--slide_dir', type=str, default='/path/to/images', help='Directory containing the original slides in .tif/.svs or other compatible format.')
    parser.add_argument('--annotations_dir', type=str, default='/path/to/annotations', help='Directory containing the annotations in .xml format.')
    parser.add_argument('--shape_file', type=str, default='/path/to/shapes.txt', help='Path to the shape file for the dataset.')
    parser.add_argument('--mask_dir', type=str, default='/path/to/masks', help='Path to the binary mask directory. Should be in .tif format.')
    parser.add_argument('--level', type=int, default=7, help='Level to construct attention masks and compare with binary mask.')
    parser.add_argument('--type', type=str, default='positive', help='Whether to check positive, negative or all slides in the test set.')
    parser.add_argument('--patch_size', type=int, default=224, help='Size of the patches.')
    parser.add_argument('--patching_level', type=int, default=2, help='Level in which the patching was done in the WSI.')
    parser.add_argument('--inference', action='store_true', default=False, help='Whether to perform inference or not.')
    parser.add_argument('--compute_masks', action='store_true', default=False, help='Whether to compute score masks or not. Should have done inference first.')
    parser.add_argument('--compare_masks', action='store_true', default=False, help='Whether to compare score vs truth masks or not. Should have done inference and compute masks first.')
    parser.add_argument('--overlay_masks', action='store_true', default=False, help='Whether to overlay annotation mask, heatmap and slide or not. Should have done inference and compute masks first.')
    parser.add_argument('--exp_config', default=None)
    args = parser.parse_args()

    main(args)