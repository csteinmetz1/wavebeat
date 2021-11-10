import os
import glob
import torch
import torchaudio
import pytorch_lightning as pl
from argparse import ArgumentParser

from wavebeat.dstcn import dsTCNModel

def beatTracker(inputFile, ckpt_dir='checkpoints/', use_gpu=False):
    """ Functional beat tracker interface. 

    Args:
        inputFile (str): Path to a valid audio file. 
        ckpt_dir (str, optional): Path to a directory containing the checkpoint. (Default: 'checkpoints/')     
        use_gpu (bool, optional): Perform inference on GPU is available. (Default: False)
        
    Returns:
        beats (ndarray): Location of predicted beats in seconds.
        downbeats (ndarray): Location of predicted downbeats in seconds.

    """

    # find the checkpoint path
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if len(ckpts) < 1:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}. See the README for details.")
    else:
        ckpt_path = ckpts[-1]

    # construct the model, and load weights from checkpoint
    model = dsTCNModel.load_from_checkpoint(ckpt_path)

    # set model to eval mode
    model.eval()

    # get the locations of the beats and downbeats
    beats, downbeats = model.predict_beats(inputFile, use_gpu=use_gpu)

    return beats, downbeats