import os
import sys
import glob
import torch 
import julius
import random
import torchaudio
import numpy as np
import scipy.signal
from tqdm import tqdm
import soxbindings as sox 

torchaudio.set_audio_backend("sox_io")

class DownbeatDataset(torch.utils.data.Dataset):
    """ Downbeat Dataset. """
    def __init__(self, 
                 audio_dir, 
                 annot_dir, 
                 audio_sample_rate=44100, 
                 target_factor=256,
                 dataset="ballroom",
                 subset="train", 
                 length=16384, 
                 preload=False, 
                 half=True, 
                 fraction=1.0,
                 augment=False,
                 dry_run=False,
                 pad_mode='constant'):
        """
        Args:
            audio_dir (str): Path to the root directory containing the audio (.wav) files.
            annot_dir (str): Path to the root directory containing the annotation (.beats) files.
            audio_sample_rate (float, optional): Sample rate of the audio files. (Default: 44100)
            target_factor (float, optional): Sample rate of the audio files. (Default: 256)
            subset (str, optional): Pull data either from "train", "val", "test", or "full-train", "full-val" subsets. (Default: "train")
            dataset (str, optional): Name of the dataset to be loaded "ballroom", "beatles", "hainsworth", "rwc_popular". (Default: "ballroom")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            half (bool, optional): Store the float32 audio as float16. (Default: True)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
            augment (bool, optional): Apply random data augmentations to input audio. (Default: False)
            dry_run (bool, optional): Train on a single example. (Default: False)
            pad_mode (str, optional): Padding type for inputs 'constant', 'reflect', 'replicate' or 'circular'. (Default: 'constant')
        """
        self.audio_dir = audio_dir
        self.annot_dir = annot_dir
        self.audio_sample_rate = audio_sample_rate
        self.target_factor = target_factor
        self.target_sample_rate = audio_sample_rate / target_factor
        self.subset = subset
        self.dataset = dataset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.augment = augment
        self.dry_run = dry_run
        self.pad_mode = pad_mode
        self.dataset = dataset

        self.target_length = int(self.length / self.target_factor)
        #print(f"Audio length: {self.length}")
        #print(f"Target length: {self.target_length}")

        # first get all of the audio files
        if self.dataset in ["beatles", "rwc_popular"]:
            file_ext = "*L+R.wav"
        elif self.dataset in ["ballroom", "hainsworth"]:
            file_ext = "*.wav"
        else:
            raise ValueError(f"Invalid dataset: {self.dataset}")

        self.audio_files = glob.glob(os.path.join(self.audio_dir, "**", file_ext))
        #self.audio_files.sort() # sort the list of audio files
        random.shuffle(self.audio_files)

        if self.subset == "train":
            start = 0
            stop = int(len(self.audio_files) * 0.8)
        elif self.subset == "val":
            start = int(len(self.audio_files) * 0.8)
            stop = int(len(self.audio_files) * 0.9)
        elif self.subset == "test":
            start = int(len(self.audio_files) * 0.9)
            stop = -1
        elif self.subset in ["full-train", "full-val"]:
            start = 0
            stop = None

        # select one file for the dry run
        if self.dry_run: 
            self.audio_files = [self.audio_files[0]] * 50
            print(f"Selected 1 file for dry run.")
        else:
            # now pick out subset of audio files
            self.audio_files = self.audio_files[start:stop]
            print(f"Selected {len(self.audio_files)} files for {self.subset} set from {self.dataset} dataset.")

        self.annot_files = []
        for audio_file in self.audio_files:
            # find the corresponding annot file
            if self.dataset in ["rwc_popular", "beatles"]:
                replace = "_L+R.wav"
            elif self.dataset in ["ballroom", "hainsworth"]:
                replace = ".wav"

            filename = os.path.basename(audio_file).replace(replace, "")

            if self.dataset == "ballroom":
                self.annot_files.append(os.path.join(self.annot_dir, f"{filename}.beats"))
            elif self.dataset == "hainsworth":
                genre_dir = os.path.basename(os.path.dirname(audio_file))
                self.annot_files.append(os.path.join(self.annot_dir, genre_dir, f"{filename}.txt"))
            elif self.dataset == "beatles":
                album_dir = os.path.basename(os.path.dirname(audio_file))
                annot_file = os.path.join(self.annot_dir, album_dir, f"{filename}.txt")
                self.annot_files.append(annot_file)
            elif self.dataset == "rwc_popular":
                album_dir = os.path.basename(os.path.dirname(audio_file))
                annot_file = os.path.join(self.annot_dir, album_dir, f"{filename}.BEAT.TXT")
                self.annot_files.append(annot_file)

        self.data = [] # when preloading store audio data and metadata
        if self.preload:
            for audio_filename, annot_filename in tqdm(zip(self.audio_files, self.annot_files), 
                                                        total=len(self.audio_files), 
                                                        ncols=80):
                    audio, target, metadata = self.load_data(audio_filename, annot_filename)
                    if self.half:
                        audio = audio.half()
                        target = target.half()
                    self.data.append((audio, target, metadata))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):

        if self.preload:
            audio, target, metadata = self.data[idx]
        else:
            # get metadata of example
            audio_filename = self.audio_files[idx]
            annot_filename = self.annot_files[idx]
            audio, target, metadata = self.load_data(audio_filename, annot_filename)

        # do all processing in float32 not float16
        audio = audio.float()
        target = target.float()

        # apply augmentations 
        if self.augment: 
            audio, target = self.apply_augmentations(audio, target)

        N_audio = audio.shape[-1]   # audio samples
        N_target = target.shape[-1] # target samples

        # random crop of the audio and target if larger than desired
        if N_audio > self.length:
            audio_start = np.random.randint(0, N_audio - self.length - 1)
            audio_stop  = audio_start + self.length
            target_start = int(audio_start / self.target_factor)
            target_stop = int(audio_stop / self.target_factor)
            audio = audio[:,audio_start:audio_stop]
            target = target[:,target_start:target_stop]
            #print(f"crop: {audio.shape} {target.shape}")

        else: # pad the audio and target is shorter than desired
            pad_size = self.length - N_audio
            #print(f"audio pad: {pad_size}")
            padl = pad_size - (pad_size // 2)
            padr = pad_size // 2
            audio = torch.nn.functional.pad(audio, 
                                            (padl, padr), 
                                            mode=self.pad_mode)
            pad_size = self.target_length - N_target
            #print(f"target pad: {pad_size}")
            padl = pad_size - (pad_size // 2)
            padr = pad_size // 2
            target = torch.nn.functional.pad(target, 
                                             (padl, padr), 
                                             mode=self.pad_mode)
            #print(f"crop: {audio.shape} {target.shape}")
            
        if self.subset in ["train", "full-train"]:
            return audio, target
        elif self.subset in ["val", "test", "full-val"]:
            # this will only work with batch size = 1
            return audio, target, metadata
        else:
            raise RuntimeError(f"Invalid subset: `{self.subset}`")

    def load_data(self, audio_filename, annot_filename):
        # first load the audio file
        audio, sr = torchaudio.load(audio_filename)
        audio = audio.float()

        # resample if needed
        if sr != self.audio_sample_rate:
            audio = julius.resample_frac(audio, sr, self.audio_sample_rate)   

        # normalize all audio inputs -1 to 1
        audio /= audio.abs().max()

        # now get the annotation information
        annot = self.load_annot(annot_filename)
        beat_samples, downbeat_samples, beat_indices, time_signature = annot

        # get metadata
        genre = os.path.basename(os.path.dirname(audio_filename))

        # convert beat_samples to beat_seconds
        beat_sec = np.array(beat_samples) / self.audio_sample_rate
        downbeat_sec = np.array(downbeat_samples) / self.audio_sample_rate

        t = audio.shape[-1]/self.audio_sample_rate # audio length in sec
        N = int(t * self.target_sample_rate) + 1   # target length in samples
        target = torch.zeros(2,N)

        # now convert from seconds to new sample rate
        beat_samples = beat_sec * self.target_sample_rate
        downbeat_samples = downbeat_sec * self.target_sample_rate

        target[0,beat_samples] = 1  # first channel is beats
        target[1,downbeat_samples] = 1  # second channel is downbeats

        metadata = {
            "Filename" : audio_filename,
            "Genre" : genre,
            "Time signature" : time_signature
        }

        return audio, target, metadata

    def load_annot(self, filename):

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        
        beat_samples = [] # array of samples containing beats
        downbeat_samples = [] # array of samples containing downbeats (1)
        beat_indices = [] # array of beat type one-hot encoded  
        time_signature = None # estimated time signature (only 3/4 or 4/4)

        for line in lines:
            if self.dataset == "ballroom":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "beatles":
                line = line.strip('\n')
                line = line.replace('\t', ' ')
                line = line.replace('  ', ' ')
                time_sec, beat = line.split(' ')
            elif self.dataset == "hainsworth":
                line = line.strip('\n')
                time_sec, beat = line.split(' ')
            elif self.dataset == "rwc_popular":
                line = line.strip('\n')
                line = line.split('\t')

                time_sec = int(line[0]) / 100.0
                beat = 1 if int(line[2]) == 384 else 2

            # convert beat to one-hot
            beat = int(beat)
            if beat == 1:
                beat_one_hot = [1,0,0,0]
            elif beat == 2:
                beat_one_hot = [0,1,0,0]
            elif beat == 3:
                beat_one_hot = [0,0,1,0]    
            elif beat == 4:
                beat_one_hot = [0,0,0,1]

            # convert seconds to samples
            beat_time_samples = int(float(time_sec) * (self.audio_sample_rate))

            beat_samples.append(beat_time_samples)
            beat_indices.append(beat)

            if beat == 1:
                downbeat_time_samples = int(float(time_sec) * (self.audio_sample_rate))
                downbeat_samples.append(downbeat_time_samples)

        # guess at the time signature
        if np.max(beat_indices) == 2:
            time_signature = "2/4"
        elif np.max(beat_indices) == 3:
            time_signature = "3/4"
        elif np.max(beat_indices) == 4:
            time_signature = "4/4"
        else:
            time_signature = "?"

        return beat_samples, downbeat_samples, beat_indices, time_signature

    def apply_augmentations(self, audio, target):

        # random gain from 0dB to -6 dB
        #if np.random.rand() < 0.2:      
        #    #sgn = np.random.choice([-1,1])
        #    audio = audio * (10**((-1 * np.random.rand() * 6)/20))   

        # phase inversion
        if np.random.rand() < 0.5:      
            audio = -audio                              

        # drop continguous frames
        if np.random.rand() < 0.05:     
            zero_size = int(self.length*0.1)
            start = np.random.randint(audio.shape[-1] - zero_size - 1)
            stop = start + zero_size
            audio[:,start:stop] = 0
            target[:,start:stop] = 0

        # apply time stretching
        if np.random.rand() < 0.3:
            factor = np.random.normal(1.0, 0.5)  
            factor = np.clip(factor, a_min=0.6, a_max=1.8)

            tfm = sox.Transformer()        

            if abs(factor - 1.0) <= 0.1: # use stretch
                tfm.stretch(1/factor)
            else:   # use tempo
                tfm.tempo(factor, 'm')

            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

            # now we update the targets based on new tempo
            dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)
            dbeat_sec = dbeat_ind / self.target_sample_rate
            new_dbeat_sec = (dbeat_sec / factor).squeeze()
            new_dbeat_ind = (new_dbeat_sec * self.target_sample_rate).long()

            beat_ind = (target[0,:] == 1).nonzero(as_tuple=False)
            beat_sec = beat_ind / self.target_sample_rate
            new_beat_sec = (beat_sec / factor).squeeze()
            new_beat_ind = (new_beat_sec * self.target_sample_rate).long()

            # now convert indices back to target vector
            new_size = int(target.shape[-1] / factor)
            streteched_target = torch.zeros(2,new_size)
            streteched_target[0,new_beat_ind] = 1
            streteched_target[1,new_dbeat_ind] = 1
            target = streteched_target

        if np.random.rand() < 0.0:
            # this is the old method (shift all beats)
            max_shift = int(0.070 * self.target_sample_rate)
            shift = np.random.randint(0, high=max_shift)
            direction = np.random.choice([-1,1])
            target = torch.roll(target, shift * direction)

        # shift targets forward/back max 70ms
        if np.random.rand() < 0.8:      
            
            # in this method we shift each beat and downbeat by a random amount
            max_shift = int(0.070 * self.target_sample_rate)

            beat_ind = torch.logical_and(target[0,:] == 1, target[1,:] != 1).nonzero(as_tuple=False) # all beats EXCEPT downbeats
            dbeat_ind = (target[1,:] == 1).nonzero(as_tuple=False)

            # shift just the downbeats
            dbeat_shifts = torch.normal(0.0, max_shift/2, size=(1,dbeat_ind.shape[-1]))
            dbeat_ind += dbeat_shifts.long()

            # now shift the non-downbeats 
            beat_shifts = torch.normal(0.0, max_shift/2, size=(1,beat_ind.shape[-1]))
            beat_ind += beat_shifts.long()

            # ensure we have no beats beyond max index
            beat_ind = beat_ind[beat_ind < target.shape[-1]]
            dbeat_ind = dbeat_ind[dbeat_ind < target.shape[-1]]  

            # now convert indices back to target vector
            shifted_target = torch.zeros(2,target.shape[-1])
            shifted_target[0,beat_ind] = 1
            shifted_target[0,dbeat_ind] = 1 # set also downbeats on first channel
            shifted_target[1,dbeat_ind] = 1

            target = shifted_target
    
        # apply pitch shifting
        if np.random.rand() < 0.5:
            sgn = np.random.choice([-1,1])
            factor = sgn * np.random.rand() * 8.0     
            tfm = sox.Transformer()        
            tfm.pitch(factor)
            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.audio_sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply a lowpass filter
        if np.random.rand() < 0.25:
            cutoff = (np.random.rand() * 4000) + 4000
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="lowpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a highpass filter
        if np.random.rand() < 0.25:
            cutoff = (np.random.rand() * 1000) + 20
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="highpass", 
                                      fs=self.audio_sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # add white noise
        if np.random.rand() < 0.05:
            wn = (torch.rand(audio.shape) * 2) - 1
            g = 10**(-(np.random.rand() * 20) - 12)/20
            audio = audio + (g * wn)

        # apply nonlinear distortion 
        if np.random.rand() < 0.2:   
            g = 10**((np.random.rand() * 12)/20)   
            audio = torch.tanh(audio)    
        
        # normalize the audio
        audio /= audio.float().abs().max()

        return audio, target