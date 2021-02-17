import os
import torch
import torchaudio
import numpy as np
import torchsummary
import pytorch_lightning as pl
from argparse import ArgumentParser

from beat.utils import center_crop, causal_crop

class Base(pl.LightningModule):
    """ Base module with train and validation loops.

        Args:
            nparams (int): Number of conditioning parameters.
            lr (float, optional): Learning rate. Default: 3e-4
            train_loss (str, optional): Training loss function from ['l1', 'stft', 'l1+stft']. Default: 'l1+stft'
            save_dir (str): Path to save audio examples from validation to disk. Default: None
            num_examples (int, optional): Number of evaluation audio examples to log after each epochs. Default: 4
        """
    def __init__(self, 
                    lr = 3e-4, 
                    save_dir = None,
                    num_examples = 4,
                    **kwargs):
        super(Base, self).__init__()
        self.save_hyperparameters()

        # these lines need to be commented out when trying
        # to jit these models in `export.py`
        self.l1      = torch.nn.L1Loss()

    def forward(self, x, p):
        pass

    @torch.jit.unused   
    def training_step(self, batch, batch_idx):
        input, target = batch

        # pass the input thrgouh the mode
        pred = self(input)

        # crop the input and target signals
        if self.hparams.causal:
            target = causal_crop(target, pred.shape[-1])
        else:
            target = center_crop(target, pred.shape[-1])

        # compute the error using appropriate loss      
        loss = self.l1(pred, target)

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target = batch

        # pass the input thrgouh the mode
        pred = self(input)

        # crop the input and target signals
        if self.hparams.causal:
            input_crop = causal_crop(input, pred.shape[-1])
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            input_crop = center_crop(input, pred.shape[-1])
            target_crop = center_crop(target, pred.shape[-1])

        # compute the validation error using all losses
        l1_loss      = self.l1(pred, target_crop)

        aggregate_loss = l1_loss

        self.log('val_loss', aggregate_loss)
        self.log('val_loss/L1', l1_loss)

        # move tensors to cpu for logging
        outputs = {
            "input" : input_crop.cpu().numpy(),
            "target": target_crop.cpu().numpy(),
            "pred"  : pred.cpu().numpy()}

        return outputs

    @torch.jit.unused
    def validation_epoch_end(self, validation_step_outputs):
        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : []}

        for out in validation_step_outputs:
            for key, val in out.items():
                bs = val.shape[0]
                for bidx in np.arange(bs):
                    outputs[key].append(val[bidx,...])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()

            # log audio examples
            self.logger.experiment.add_audio(f"input/{idx}",  
                                             i, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            self.logger.experiment.add_audio(f"target/{idx}", 
                                             t, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)
            self.logger.experiment.add_audio(f"pred/{idx}",   
                                             p, self.global_step, 
                                             sample_rate=self.hparams.sample_rate)

            if self.hparams.save_dir is not None:
                if not os.path.isdir(self.hparams.save_dir):
                    os.makedirs(self.hparams.save_dir)

                input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    torch.tensor(i).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    torch.tensor(t).view(1,-1).float(),
                                    sample_rate=self.hparams.sample_rate)

                torchaudio.save(os.path.join(self.hparams.save_dir, 
                                f"{idx}-pred-{self.hparams.train_loss}-{int(prm[0]):1d}-{prm[1]:0.2f}.wav"), 
                                torch.tensor(p).view(1,-1).float(),
                                sample_rate=self.hparams.sample_rate)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-3)
        # --- vadliation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser