import os
import glob
import torch
import torchaudio
import pytorch_lightning as pl
from argparse import ArgumentParser

from wavebeat.dstcn import dsTCNModel

parser = ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--model', type=str, default="checkpoints/")

args = parser.parse_args()

# find the checkpoint path
ckpts = glob.glob(os.path.join(args.model, "*.ckpt"))
if len(ckpts) < 1:
    raise RuntimeError(f"No checkpoints found in {args.model}.")
else:
    ckpt_path = ckpts[-1]

# construct the model, and load weights from checkpoint
model = dsTCNModel.load_from_checkpoint(ckpt_path)

# set model to eval mode
model.eval()

# get the locations of the beats and downbeats
beats, downbeats = model.predict_beats(args.input)

# print some results to terminal
print(f"Beats found in {args.input}")
print("-" * 32)
for beat in beats:
    print(f"{beat:0.2f}")

print()
print(f"Downbeats found in {args.input}")
print("-" * 32)
for downbeat in downbeats:
    print(f"{downbeat:0.2f}")