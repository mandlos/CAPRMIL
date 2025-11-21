import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import sys
import pandas as pd
import h5py
import math
from math import floor
from PIL import Image
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.wsi_dataset import Wsi_Region
from utils.utils import *
from utils.transform_utils import get_eval_transforms
from utils.file_utils import save_hdf5
from utils.constants import MODEL2CONSTANTS
from src.model_transmil import TransMIL
from src.model_clam import CLAM_SB, CLAM_MB
from src.gp_models import NNGP
from wsi_core.WholeSlideImage import WholeSlideImage

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 16

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size'] 
    
    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=num_workers)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        
        with torch.inference_mode():
            features = feature_extractor(roi)
            print(f'\nCompute from patches, shape of features: {features.shape}\n')
            if attn_save_path is not None:
                print(f'\n attn save path not None\n')
                if isinstance(model, TransMIL):
                    A = model(features)['A']
                elif isinstance(model, NNGP):                    
                    A = model(features.unsqueeze(0))['attention'].mean(dim=0)
                    print(f'\nCompute from patches, shape of A: {A.shape}\n')
                elif isinstance(model, CLAM_SB) or isinstance(model, CLAM_MB):
                    A = model(features, attention_only=True)
                    if A.size(0) > 1: #CLAM multi-branch attention
                        A = A[clam_pred]
                    
                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def compute_from_patches_custom(wsi_object, img_transforms, feature_extractor=None, 
                                clam_pred=None, model=None, batch_size=512,  
                                attn_save_path=None, attention_type:str=None, ref_scores=None, 
                                feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size'] 
    
    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=num_workers)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    if not os.path.exists(feat_save_path):
        for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
            roi = roi.to(device)
            
            with torch.inference_mode():
                features = feature_extractor(roi)
                print(f'\nCompute from patches, shape of features: {features.shape}\n')
            if feat_save_path is not None:
                asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
                save_hdf5(feat_save_path, asset_dict, mode=mode)

            mode = "a"
            del features

    with h5py.File(feat_save_path, 'r') as f:
        file = h5py.File(feat_save_path, 'r')
        f = file['features'][:]
        c = file['coords'][:]
        features = torch.from_numpy(f).to(device)
        coords = torch.from_numpy(c).to(device)

    # Read features file and do compute attentions
    assert attn_save_path is not None, 'attn_save_path is None'
    with torch.inference_mode():
        if isinstance(model, TransMIL):
            A = model(features)['A']
        elif isinstance(model, NNGP):
            if attention_type == 'mean':
                A = model(features.unsqueeze(0))['attention'].mean(dim=0)
            elif attention_type == 'std':
                A = model(features.unsqueeze(0))['attention'].std(dim=0)
            elif attention_type == 'mean/std':
                eps = 1.e-2
                A = model(features.unsqueeze(0))['attention']
                A_mean = A.mean(dim=0)
                A_mean = (A_mean - A_mean.min()) / (A_mean.max() - A_mean.min())
                A_std = A.std(dim=0)
                A_std = (A_std - A_std.min()) / (A_std.max() - A_std.min())
                A = A_mean / (A_std + eps)
            else:
                raise ValueError(f'Invalid attention type: {attention_type}')
        elif isinstance(model, CLAM_SB) or isinstance(model, CLAM_MB):
            A = model(features, attention_only=True)
            if A.size(0) > 1: #CLAM multi-branch attention
                A = A[clam_pred]
        
    A = (A - A.min()) / (A.max() - A.min())
    A = A.view(-1, 1).cpu().numpy()

    if ref_scores is not None:
        for score_idx in range(len(A)):
            A[score_idx] = score2percentile(A[score_idx], ref_scores)

    asset_dict = {'attention_scores': A, 'coords': coords.cpu().numpy()}
    save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    return attn_save_path, feat_save_path, wsi_object
