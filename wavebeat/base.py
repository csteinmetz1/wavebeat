import os
import torch
import julius
import madmom
import mir_eval
import torchaudio
import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser

from wavebeat.plot import plot_activations, make_table, plot_histogram
from wavebeat.loss import GlobalMSELoss, GlobalBCELoss, BCFELoss
from wavebeat.utils import center_crop, causal_crop
from wavebeat.eval import evaluate, find_beats
from wavebeat.filter import FIRFilter

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
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()
        self.gmse = GlobalMSELoss()
        self.gbce = GlobalBCELoss()
        self.bcfe = BCFELoss()

    def forward(self, x):
        pass

    @torch.jit.unused  
    def predict_beats(self, filename, use_gpu=False):
        """ Load and audio file and predict the beat and downbeat loctions. 
        
        Args:
            filename (str): Path to an audio file. 
            use_gpu (bool, optional): Perform inference on GPU is available. 
        
        Returns:
            beats (ndarray): Location of predicted beats in seconds.
            downbeats (ndarray): Location of predicted downbeats in seconds.
        """
        
        # load the audio into tensor
        audio, sr = torchaudio.load(filename)

        # resample to 22.05 kHz if needed
        if sr != self.hparams.audio_sample_rate:
            audio = julius.resample_frac(audio, sr, self.hparams.audio_sample_rate)   

        if audio.shape[0] > 1:
            print("Loaded multichannel audio. Summing to mono...")
            audio = audio.mean(dim=0, keepdim=True)

        # normalize the audio
        audio /= audio.abs().max()

        # add a batch dim
        audio = audio.unsqueeze(0)

        if use_gpu:
            audio = audio.to("cuda:0")
            self.to("cuda:0")
        else:
            self.to('cpu')

        # pass audio to model
        with torch.no_grad():
            pred = torch.sigmoid(self(audio))

        # move data back to CPU
        if use_gpu:
            pred = pred.cpu()

        # separate the beats and downbeat activations
        p_beats = pred[0,0,:]
        p_downbeats = pred[0,1,:]

        # use peak picking to find locations of beats and downbeats
        _, beats, _ = find_beats(p_beats.numpy(), 
                                    p_beats.numpy(), 
                                    beat_type="beat",
                                    sample_rate=self.hparams.target_sample_rate)

        _, downbeats, _ = find_beats(p_downbeats.numpy(), 
                                        p_downbeats.numpy(), 
                                        beat_type="downbeat",
                                        sample_rate=self.hparams.target_sample_rate)

        return beats, downbeats

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
        #loss, _, _ = self.gbce(pred, target)
        loss, _, _ = self.bcfe(pred, target)

        self.log('train_loss', 
                 loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True)

        return loss

    @torch.jit.unused
    def validation_step(self, batch, batch_idx):
        input, target, metadata = batch

        # pass the input thrgouh the mode
        pred = self(input)

        # crop the input and target signals
        if self.hparams.causal:
            target_crop = causal_crop(target, pred.shape[-1])
        else:
            target_crop = center_crop(target, pred.shape[-1])

        # compute the validation error using all losses
        #loss, _, _ = self.gbce(pred, target_crop)
        loss, _, _ = self.bcfe(pred, target_crop)

        self.log('val_loss', loss)

        # apply sigmoid after computing loss
        pred = torch.sigmoid(pred)

        # move tensors to cpu for logging
        outputs = {
            "input" : input.cpu(),
            "target": target_crop.cpu(),
            "pred"  : pred.cpu(),
            "Filename" : metadata['Filename'],
            "Genre" : metadata['Genre'],
            "Time signature" : metadata['Time signature']
            }

        return outputs

    @torch.jit.unused
    def validation_epoch_end(self, validation_step_outputs):
        # flatten the output validation step dicts to a single dict
        outputs = {
            "input" : [],
            "target" : [],
            "pred" : [],
            "Filename" : [],
            "Genre" : [],
            "Time signature" : []}

        metadata_keys = ["Filename", "Genre", "Time signature"]

        for out in validation_step_outputs:
            for key, val in out.items():
                if key not in metadata_keys:
                    bs = val.shape[0]
                else:
                    bs = len(val)
                for bidx in np.arange(bs):
                    if key not in metadata_keys:
                        outputs[key].append(val[bidx,...])
                    else:
                        outputs[key].append(val[bidx])

        example_indices = np.arange(len(outputs["input"]))
        rand_indices = np.random.choice(example_indices,
                                        replace=False,
                                        size=np.min([len(outputs["input"]), self.hparams.num_examples]))

        # compute metrics 
        songs = []
        beat_f1_scores = []
        downbeat_f1_scores = []
        #dbn_beat_f1_scores = []
        #dbn_downbeat_f1_scores = []
        for idx in np.arange(len(outputs["input"])):
            t = outputs["target"][idx].squeeze()
            p = outputs["pred"][idx].squeeze()
            f = outputs["Filename"][idx]
            g = outputs["Genre"][idx]
            s = outputs["Time signature"][idx]

            beat_scores, downbeat_scores = evaluate(p, t, self.hparams.target_sample_rate)

            songs.append({
                "Filename" : f,
                "Genre" : g,
                "Time signature" : s,
                "Beat F-measure" : beat_scores['F-measure'],
                "Downbeat F-measure" : downbeat_scores['F-measure'],
                #"(DBN) Beat F-measure" : dbn_beat_scores['F-measure'],
                #"(DBN) Downbeat F-measure" : dbn_downbeat_scores['F-measure']
            })

            beat_f1_scores.append(beat_scores['F-measure'])
            downbeat_f1_scores.append(downbeat_scores['F-measure'])
            #dbn_beat_f1_scores.append(dbn_beat_scores['F-measure'])
            #dbn_downbeat_f1_scores.append(dbn_downbeat_scores['F-measure'])

        beat_f_measure = np.mean(beat_f1_scores)
        downbeat_f_measure = np.mean(downbeat_f1_scores)
        self.log('val_loss/Beat F-measure', torch.tensor(beat_f_measure))
        self.log('val_loss/Downbeat F-measure', torch.tensor(downbeat_f_measure))
        self.log('val_loss/Joint F-measure', torch.tensor(np.mean([beat_f_measure,downbeat_f_measure])))
        #self.log('val_loss/(DBN) Beat F-measure', np.mean(dbn_beat_f1_scores))
        #self.log('val_loss/(DBN) Downbeat F-measure', np.mean(dbn_downbeat_f1_scores))

        self.logger.experiment.add_text("perf", 
                                        make_table(songs),
                                        self.global_step)
    
        # log score histograms plots
        self.logger.experiment.add_image(f"hist/F-measure",
                                         plot_histogram(songs),
                                         self.global_step)
 
        for idx, rand_idx in enumerate(list(rand_indices)):
            i = outputs["input"][rand_idx].squeeze()
            t = outputs["target"][rand_idx].squeeze()
            p = outputs["pred"][rand_idx].squeeze()
            f = outputs["Filename"][idx]
            g = outputs["Genre"][idx]
            s = outputs["Time signature"][idx]

            t_beats = t[0,:]
            t_downbeats = t[1,:]
            p_beats = p[0,:]
            p_downbeats = p[1,:]

            ref_beats, est_beats, est_sm = find_beats(t_beats.numpy(), 
                                                      p_beats.numpy(), 
                                                      beat_type="beat",
                                                      sample_rate=self.hparams.target_sample_rate)

            ref_downbeats, est_downbeats, est_downbeat_sm = find_beats(t_downbeats.numpy(), 
                                                                       p_downbeats.numpy(), 
                                                                       beat_type="downbeat",
                                                                       sample_rate=self.hparams.target_sample_rate)
            # log audio examples
            self.logger.experiment.add_audio(f"input/{idx}",  
                                             i, self.global_step, 
                                             sample_rate=self.hparams.audio_sample_rate)

            # log beats plots
            self.logger.experiment.add_image(f"act/{idx}",
                                             plot_activations(ref_beats, 
                                                              est_beats, 
                                                              est_sm,
                                                              self.hparams.target_sample_rate,
                                                              ref_downbeats=ref_downbeats,
                                                              est_downbeats=est_downbeats,
                                                              est_downbeats_sm=est_downbeat_sm,
                                                              song_name=f),
                                             self.global_step)

            if self.hparams.save_dir is not None:
                if not os.path.isdir(self.hparams.save_dir):
                    os.makedirs(self.hparams.save_dir)

                input_filename = os.path.join(self.hparams.save_dir, f"{idx}-input-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")
                target_filename = os.path.join(self.hparams.save_dir, f"{idx}-target-{int(prm[0]):1d}-{prm[1]:0.2f}.wav")

                if not os.path.isfile(input_filename):
                    torchaudio.save(input_filename, 
                                    torch.tensor(i).view(1,-1).float(),
                                    sample_rate=self.hparams.audio_sample_rate)

                if not os.path.isfile(target_filename):
                    torchaudio.save(target_filename,
                                    torch.tensor(t).view(1,-1).float(),
                                    sample_rate=self.hparams.audio_sample_rate)

                torchaudio.save(os.path.join(self.hparams.save_dir, 
                                f"{idx}-pred-{self.hparams.train_loss}-{int(prm[0]):1d}-{prm[1]:0.2f}.wav"), 
                                torch.tensor(p).view(1,-1).float(),
                                sample_rate=self.hparams.audio_sample_rate)

    @torch.jit.unused
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @torch.jit.unused
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    @torch.jit.unused
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        #                                                          patience=self.hparams.patience, 
        #                                                          verbose=True,
        #                                                          mode='max')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                       step_size=10, 
                                                       gamma=0.5,
                                                       verbose=True)                                                      
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss/Joint F-measure'
        }

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- training related ---
        parser.add_argument('--lr', type=float, default=1e-2)
        # --- vadliation related ---
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--num_examples', type=int, default=4)

        return parser