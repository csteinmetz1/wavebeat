import os
import glob
import torch
import torchsummary
from itertools import product
import pytorch_lightning as pl
from argparse import ArgumentParser

from beat.tcn import TCNModel
from beat.dstcn import dsTCNModel
from beat.lstm import LSTMModel
from beat.waveunet import WaveUNetModel
from beat.data import DownbeatDataset

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--model_type', type=str, default='tcn', help='tcn, lstm, waveunet, or dstcn')
parser.add_argument('--dataset', type=str, default='ballroom')
parser.add_argument('--beatles_audio_dir', type=str, default='./data')
parser.add_argument('--beatles_annot_dir', type=str, default='./data')
parser.add_argument('--ballroom_audio_dir', type=str, default='./data')
parser.add_argument('--ballroom_annot_dir', type=str, default='./data')
parser.add_argument('--hainsworth_audio_dir', type=str, default='./data')
parser.add_argument('--hainsworth_annot_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
parser.add_argument('--preload', action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=44100)
parser.add_argument('--target_factor', type=int, default=256)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=65536)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--dry_run', action='store_true')

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# let the model add what it wants
if temp_args.model_type == 'tcn':
    parser = TCNModel.add_model_specific_args(parser)
elif temp_args.model_type == 'lstm':
    parser = LSTMModel.add_model_specific_args(parser)
elif temp_args.model_type == 'waveunet':
    parser = WaveUNetModel.add_model_specific_args(parser)
elif temp_args.model_type == 'dstcn':
    parser = dsTCNModel.add_model_specific_args(parser)

# parse them args
args = parser.parse_args()

datasets = ["beatles", "ballroom", "hainsworth", "rwc_popular"]

for val_set in datasets:

    train_sets = [d for d in datasets if d != val_set]

    print(f"Training with {train_sets[0]}, {train_sets[1]}, and {train_sets[2]}.")
    print(f"Validating with {val_set}.")

    # set the seed
    pl.seed_everything(42)

    #
    args.default_root_dir = os.path.join("lightning_logs", "cv", val_set)
    print(args.default_root_dir)

    # create the trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # setup the dataloaders
    train_datasets = []
    for train_set in train_sets:
        if train_set == "beatles":
            audio_dir = args.beatles_audio_dir
            annot_dir = args.beatles_annot_dir
        elif train_set == "ballroom":
            audio_dir = args.ballroom_audio_dir
            annot_dir = args.ballroom_annot_dir
        elif train_set == "hainsworth":
            audio_dir = args.hainsworth_audio_dir
            annot_dir = args.hainsworth_annot_dir
        elif train_set == "rwc_popular":
            audio_dir = args.rwc_popular_audio_dir
            annot_dir = args.rwc_popular_annot_dir

        train_dataset = DownbeatDataset(audio_dir,
                                        annot_dir,
                                        dataset=train_set,
                                        audio_sample_rate=args.audio_sample_rate,
                                        target_factor=args.target_factor,
                                        subset="full-train",
                                        fraction=args.train_fraction,
                                        augment=args.augment,
                                        half=True if args.precision == 16 else False,
                                        preload=args.preload,
                                        length=args.train_length,
                                        dry_run=args.dry_run)
        train_datasets.append(train_dataset)

    train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
    train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                                   shuffle=args.shuffle,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True)
    if val_set == "beatles":
        audio_dir = args.beatles_audio_dir
        annot_dir = args.beatles_annot_dir
    elif val_set == "ballroom":
        audio_dir = args.ballroom_audio_dir
        annot_dir = args.ballroom_annot_dir
    elif val_set == "hainsworth":
        audio_dir = args.hainsworth_audio_dir
        annot_dir = args.hainsworth_annot_dir
    elif val_set == "rwc_popular":
        audio_dir = args.rwc_popular_audio_dir
        annot_dir = args.rwc_popular_annot_dir

    val_dataset = DownbeatDataset(audio_dir,
                                annot_dir,
                                dataset=val_set,
                                audio_sample_rate=args.audio_sample_rate,
                                target_factor=args.target_factor,
                                subset="full-val",
                                augment=False,
                                half=True if args.precision == 16 else False,
                                preload=args.preload,
                                length=args.eval_length,
                                dry_run=args.dry_run)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                shuffle=args.shuffle,
                                                batch_size=4,
                                                num_workers=args.num_workers,
                                                pin_memory=False)

    # create the model with args
    dict_args = vars(args)
    dict_args["nparams"] = 2
    dict_args["target_sample_rate"] = args.audio_sample_rate / args.target_factor

    if args.model_type == 'tcn':
        model = TCNModel(**dict_args)
        rf = model.compute_receptive_field()
        print(f"Model has receptive field of {(rf/args.sample_rate)*1e3:0.1f} ms ({rf}) samples")
    elif args.model_type == 'lstm':
        model = LSTMModel(**dict_args)
    elif args.model_type == 'waveunet':
        model = WaveUNetModel(**dict_args)
    elif args.model_type == 'dstcn':
        model = dsTCNModel(**dict_args)

    # summary 
    torchsummary.summary(model, [(1,args.train_length)], device="cpu")

    # train!
    trainer.fit(model, train_dataloader, val_dataloader)
