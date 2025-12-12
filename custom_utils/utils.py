import os
import pathlib
import ftplib
from ftplib import FTP
import urllib
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET
import openslide
import time
from pytorch_lightning.callbacks import Callback

def print_directory_structure(directory, indent=0, printout_lim:int = 10):
    """
    Recursively prints the structure of a directory.
    
    Args:
        directory (str): The path to the directory.
        indent (int): The current indentation level.
        printout_lim (int): Maximum number of files to be displayed for a directory
    """
    print("|   " * indent + "|--", os.path.basename(directory))
    if os.path.isdir(directory):
        for item in os.listdir(directory)[:printout_lim]:
            item_path = os.path.join(directory, item)
            print_directory_structure(item_path, indent + 1, printout_lim)


def _get_remote_directory_size(ftp, directory):
    total_size = 0
    try:
        ftp.cwd(directory)
        files = []
        ftp.retrlines('LIST', files.append)
        for line in files:
            tokens = line.split(None, 8)
            if tokens[0].startswith('-'):  # Regular file
                total_size += int(tokens[4])
            elif tokens[0].startswith('d'):  # Directory
                sub_directory = directory + '/' + tokens[-1]
                total_size += _get_remote_directory_size(ftp, sub_directory)
    except Exception as e:
        print(e)
    return total_size

def check_remote_directory_size(url):
    '''
    Returns the size of the remote directory in bytes

    Args:
        url (str): URL of the remote directory
    Returns:
        int: Size of the remote directory
    '''
    
    parts = url.split('/')
    hostname = parts[2]
    remote_path = '/' + '/'.join(parts[3:])

    with FTP(hostname) as ftp:
        ftp.login()
        size = _get_remote_directory_size(ftp, remote_path)
        return size



def check_remote_directory_size(url):
    '''
    Returns the size of the remote directory in bytes

    Args:
        url (str): URL of the remote directory
    Returns:
        int: Size of the remote directory
    '''
    
    parts = url.split('/')
    hostname = parts[2]
    remote_path = '/' + '/'.join(parts[3:])

    with FTP(hostname) as ftp:
        ftp.login()
        size = get_remote_directory_size(ftp, remote_path)
        return size

def _download_ftp_directory(ftp, remote_path, local_path):
    try:
        os.makedirs(local_path)
    except FileExistsError:
        pass
    
    # Change directory
    ftp.cwd(remote_path)
    # List of contents
    filenames = ftp.nlst()
    desc = f"Navigating {remote_path}"

    for filename in tqdm(filenames, desc=desc, position=0):
        local_fpath = os.path.join(local_path, filename)
        remote_fpath = os.path.join(remote_path, filename)
        
        try:
            ftp.cwd(remote_fpath)
            logging.info(f"Navigating {remote_fpath}...")
            _download_ftp_directory(ftp, remote_fpath, local_fpath)
        except Exception as e:
            logging.info(f"Downloading {filename}...")
            with open(local_fpath, 'wb') as f:
                with tqdm(total=ftp.size(filename), 
                          desc=f"Downloading {filename}", 
                          position=0, unit='%', 
                          bar_format="{desc}: {percentage:.0f}% ({n:.0f}/{total:.0f})") as pbar:
                    def callback(data):
                        f.write(data)
                        pbar.update(len(data))
                        
                    ftp.retrbinary('RETR ' + filename, callback)
            logging.info(f"{filename} downloaded.")

def download_ftp(url, local_path):
    parts = url.split('/')
    hostname = parts[2]
    remote_path = '/' + '/'.join(parts[3:])
    
    with FTP(hostname) as ftp:
        ftp.login()
        _download_ftp_directory(ftp, remote_path, local_path)



# Function to plot together multiple jpg or png images
def plot_images(images, labels, save_name=None, save=False, show=True):
    '''
    Function to plot multiple images side by side.
    images: list of strings containing the paths to the images
    labels: list of strings containing the labels of the images
    save_name: string containing the name of the file to save the plot
    '''
    fig, ax = plt.subplots(figsize=(10*len(images), 10), ncols=len(images), sharex=True, sharey=True, gridspec_kw={'wspace': 0})
    if len(images)>1:
        for i, image in enumerate(images):
            img = plt.imread(image)
            ax[i].imshow(img)
            ax[i].axis('off')
            ax[i].set_title(labels[i])
    else:
        img = plt.imread(images[0])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(labels[0])
        
    if save: fig.savefig(save_name)
    if not show: 
        plt.close(fig)
    else:
        plt.show(fig)


# Function to plot receiver operating characteristic curve
def roc_plot(data=[], labels=None, save_name=[], save=False, lineplot_args={}, show=True, legend_title='Feature Extractor'):
    '''
    Function to plot ROC curves for multiple models.
    data: pandas DataFrame containing false positive and true positive rates for each model
    labels: list of strings containing the names of the models
    save_name: string containing the name of the file to save the plot
    '''
    fig, ax = plt.subplots(figsize=(5, 5))

    major_tick_params = {'direction':'in', 'left':True, 'right':True, 'top':True, 'bottom':True, 'gridOn':True, 'which':'major', 'pad':10}
    minor_tick_params = {'direction':'in', 'left':True, 'right':True, 'top':True, 'bottom':True, 'gridOn':False, 'which':'minor', 'pad':10}
    legend_params = {'frameon':True, 'loc':'lower right', 'fontsize':'small', 'title':legend_title}

    for index, df in enumerate(data):
        if labels is not None:
            sns.lineplot(data=df, x='fpr', y='tpr', ax=ax, label=labels[index], **lineplot_args)
        else:
            sns.lineplot(data=df, x='fpr', y='tpr', ax=ax, **lineplot_args)

    # Draw y=x line with sns.lineplot
    sns.lineplot(x=[0,1], y=[0,1], ax=ax, color='gray', linestyle='--')

    # x and y axis
    epsilon = 0.00001
    ax.set_xlim(0-epsilon,1+epsilon)
    ax.set_ylim(0-epsilon,1+epsilon)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Ticks
    ax.minorticks_on()
    ax.xaxis.set_tick_params(**major_tick_params)
    ax.xaxis.set_tick_params(**minor_tick_params)
    ax.yaxis.set_tick_params(**major_tick_params)
    ax.yaxis.set_tick_params(**minor_tick_params)
    ax.legend(**legend_params)
    ax.title.set_text('ROC')
    fig.tight_layout(pad=2)

    if save: fig.savefig(save_name)
    if not show: 
        plt.close(fig)
    else:
        plt.show(fig)

# Helper function to parse XML annotations
def parse_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for annotation in root.findall('.//Annotation'):
        for coordinate in annotation.findall('.//Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            annotations.append((x, y))
    return annotations

# Helper function to draw annotations
def draw_annotations(img, annotations, downsample_factor, save_name=[], title=[], show=True, save=False):
    fig, ax = plt.subplots(figsize=(15, 15), ncols=1, sharex=True, sharey=True, gridspec_kw={'wspace': 0})
    ax.imshow(img)
    # if there are annotations, plot them
    if len(annotations)>0:
        for annotation in annotations:
            x, y = zip(*annotation)
            ax.scatter([i / downsample_factor for i in x], [i / downsample_factor for i in y], s=0.1, c='yellow')
    ax.set_title(title)
    ax.axis('off')
    if save: fig.savefig(save_name)
    if show:
        plt.show(fig)
    else:
        plt.close(fig)
    
    fig.tight_layout()

def get_slide_magnification(slide_path):
    slide = openslide.OpenSlide(slide_path)
    properties = slide.properties

    slide_id = os.path.basename(slide_path)

    mag_dict = {
        'slide_id': slide_id,
        'base_magnification': None,
        'Xresolution': None,
        'Yresolution': None,
        'ResolutionUnits': None
    }

    # Check for mpp-x / mpp-y properties first
    if 'openslide.mpp-x' in properties and 'openslide.mpp-y' in properties:
        mpp_x = float(properties['openslide.mpp-x'])
        mpp_y = float(properties['openslide.mpp-y'])

        # Typical microscope magnifications correspond to approximate mpp (micron per pixel) as follows:
        # 40x ~ 0.25 mpp, 20x ~ 0.5 mpp, 10x ~ 1.0 mpp, etc.
        avg_mpp = (mpp_x + mpp_y) / 2
        if avg_mpp < 0.3:
            magnification = '40X'
        elif avg_mpp < 0.6:
            magnification = '20X'
        elif avg_mpp < 1.2:
            magnification = '10X'
        else:
            magnification = 'Unknown'
        
        mag_dict['base_magnification'] = magnification
        mag_dict['Xresolution'] = mpp_x
        mag_dict['Yresolution'] = mpp_y
        mag_dict['ResolutionUnits'] = 'microns-per-pixel'
    # If these are not present, use TIFF resolution tags
    elif 'tiff.XResolution' in properties and 'tiff.YResolution' in properties:
        xres = float(properties['tiff.XResolution'])
        yres = float(properties['tiff.YResolution'])
        res_unit = properties.get('tiff.ResolutionUnit', 'unknown')

        mag_dict['Xresolution'] = xres
        mag_dict['Yresolution'] = yres
        mag_dict['ResolutionUnits'] = res_unit
        
        # Infer magnification roughly
        # This can vary by scanner, and you might want calibration data.
        if res_unit == 'centimeter':
            mpp_x = 10000.0 / xres   # converting to microns per pixel
            mpp_y = 10000.0 / yres
            avg_mpp = (mpp_x + mpp_y) / 2
            if avg_mpp < 0.3:
                magnification = '40X'
            elif avg_mpp < 0.6:
                magnification = '20X'
            elif avg_mpp < 1.2:
                magnification = '10X'
            else:
                magnification = 'Unknown'

            mag_dict['base_magnification'] = magnification
    else:
        mag_dict['base_magnification'] = 'unknown'

    slide.close()
    return mag_dict

class EpochTimingCallback(Callback):
    def __init__(self):
        self.train_times = []
        self.val_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        duration = time.time() - self._train_start_time
        self.train_times.append(duration)
        pl_module.log("train_epoch_time", duration, prog_bar=True, on_epoch=True)
        avg_train = sum(self.train_times) / len(self.train_times)
        pl_module.log("avg_train_epoch_time", avg_train, prog_bar=True, on_epoch=True)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_start_time = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        duration = time.time() - self._val_start_time
        self.val_times.append(duration)
        pl_module.log("val_epoch_time", duration, prog_bar=True, on_epoch=True)
        avg_val = sum(self.val_times) / len(self.val_times)
        pl_module.log("avg_val_epoch_time", avg_val, prog_bar=True, on_epoch=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["train_epoch_times"] = self.train_times
        checkpoint["val_epoch_times"] = self.val_times

def main():
    return

if __name__ == "__main__":
    print("No code to execute here, only imports!")
