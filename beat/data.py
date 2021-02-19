import os
import sys
import glob
import torch 
import julius
import torchaudio
import numpy as np
import scipy.signal
import soxbindings as sox 

torchaudio.set_audio_backend("sox_io")

class BallroomDataset(torch.utils.data.Dataset):
    """ Ballroom Dataset. 

        Audio: [http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)

        Annotations: [https://github.com/CPJKU/BallroomAnnotations](https://github.com/CPJKU/BallroomAnnotations)
    
    """
    def __init__(self, 
                 audio_dir, 
                 annot_dir, 
                 sample_rate=44100, 
                 subset="train", 
                 length=16384, 
                 preload=False, 
                 half=True, 
                 fraction=1.0,
                 augment=False,
                 dry_run=False,
                 pad_mode='reflect'):
        """
        Args:
            audio_dir (str): Path to the root directory containing the audio (.wav) files.
            annot_dir (str): Path to the root directory containing the annotation (.beats) files.
            sample_rate (int, optional): Sample rate of the audio files. (Default: 44100)
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. (Default: "train")
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
        self.sample_rate = sample_rate
        self.subset = subset
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.augment = augment
        self.dry_run = dry_run
        self.pad_mode = pad_mode

        # first get all of the audio files
        self.audio_files = glob.glob(os.path.join(self.audio_dir, "**", "*.wav"))
        self.audio_files.sort() # sort the list of audio files

        if self.subset == "train":
            start = 0
            stop = int(len(self.audio_files) * 0.8)
        elif self.subset == "val":
            start = int(len(self.audio_files) * 0.8)
            stop = int(len(self.audio_files) * 0.9)
        elif self.subset == "test":
            start = int(len(self.audio_files) * 0.9)
            stop = -1
        elif self.subset == "full":
            start = 0
            stop = -1

        # select one file for the dry run
        if self.dry_run: 
            self.audio_files = [self.audio_files[0]] * 50
            print(f"Selected 1 file for dry run.")
        else:
            # now pick out subset of audio files
            self.audio_files = self.audio_files[start:stop]
            print(f"Selected {len(self.audio_files)} files for {self.subset} set.")

        self.annot_files = []
        for audio_file in self.audio_files:
            # find the corresponding annot file
            filename = os.path.basename(audio_file).replace(".wav", "")
            self.annot_files.append(os.path.join(self.annot_dir, f"{filename}.beats"))
        
        for audio, annot in zip(self.audio_files, self.annot_files):
            self.load_annot(annot)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):

        # first load the audio file
        audio, sr = torchaudio.load(self.audio_files[idx])

        # resample if needed
        if sr != self.sample_rate:
            audio = julius.resample_frac(audio, sr, self.sample_rate)   
        
        # normalize all inputs -1 to 1
        audio /= audio.abs().max()

        # now get the annotation information
        beat_samples, downbeat_samples, beat_indices = self.load_annot(self.annot_files[idx])

        # now we construct the target sequence with beat (1) and no beat (0)
        N = audio.shape[-1]
        target = torch.zeros(2,N)
        target[0,beat_samples] = 1  # first channel is beats
        target[1,downbeat_samples] = 1  # second channel is downbeats

        # apply augmentations 
        if self.augment: 
            audio, target = self.apply_augmentations(audio, target)

        # now we take a random crop of both
        if (N - self.length - 1) < 0:
            start = 0
            stop = self.length
        else:
            start = np.random.randint(N - self.length - 1)
            stop = start + self.length

        audio = audio[:,start:stop]
        target = target[:,start:stop]

        # check the length and pad if
        if audio.shape[-1] < self.length:
            pad_size = self.length - audio.shape[-1]
            pad_left = pad_size - (pad_size // 2)
            pad_right = pad_size // 2
            audio = torch.nn.functional.pad(audio.view(1,1,-1),
                                           (pad_left,pad_right),
                                           mode=self.pad_mode)
            audio = audio.view(1,-1)
        elif audio.shape[-1] > self.length:
            audio = audio[:,:self.length]

        if target.shape[-1] < self.length:
            pad_size = self.length - target.shape[-1]
            pad_left = pad_size - (pad_size // 2)
            pad_right = pad_size // 2
            target = torch.nn.functional.pad(target.view(1,2,-1),
                                             (pad_left,pad_right),
                                             mode=self.pad_mode)
            target = target.view(2,-1)
        elif target.shape[-1] > self.length:
            target = target[:,:self.length]

        metadata = {
            "filename" : self.audio_files[idx]
        }

        return audio, target, metadata

    def load_annot(self, filename):

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        
        beat_samples = [] # array of samples containing beats
        downbeat_samples = [] # array of samples containing downbeats (1)
        beat_indices = [] # array of beat type one-hot encoded  

        for line in lines:
            line = line.strip('\n')
            line = line.replace('\t', ' ')
            time_sec, beat = line.split(' ')

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
            beat_time_samples = int(float(time_sec) * (self.sample_rate))

            beat_samples.append(beat_time_samples)
            beat_indices.append(beat_one_hot)

            if beat == 1:
                downbeat_time_samples = int(float(time_sec) * (self.sample_rate))
                downbeat_samples.append(downbeat_time_samples)

        return beat_samples, downbeat_samples, beat_indices

    def apply_augmentations(self, audio, target):

        # random gain from 0dB to -6 dB
        #if np.random.rand() < 0.2:      
        #    #sgn = np.random.choice([-1,1])
        #    audio = audio * (10**((-1 * np.random.rand() * 6)/20))   

        # phase inversion
        if np.random.rand() < 0.5:      
            audio = -audio                              

        # apply nonlinear distortion 
        if np.random.rand() < 0.2:   
            g = 10**((np.random.rand() * 12)/20)   
            audio = torch.tanh(audio)    

        # drop continguous frames
        if np.random.rand() < 0.05:     
            zero_size = int(self.length*0.1)
            start = np.random.randint(audio.shape[-1] - zero_size - 1)
            stop = start + zero_size
            audio[:,start:stop] = 0
            target[:,start:stop] = 0

        # shift targets forward/back max 10ms
        if np.random.rand() < 0.4:      
            max_shift = int(0.10 * self.sample_rate)
            shift = np.random.randint(0, high=max_shift)
            direction = np.random.choice([-1,1])
            target = torch.roll(target, shift * direction)

        # apply time stretching
        #if np.random.rand() < 0.0:
        #    sgn = np.random.choice([-1,1])
        #    factor = sng * np.random.rand() * 0.2     
        #    tfm = sox.Transformer()        
        #    tfm.tempo(factor, 'm')
        #    audio = tfm.build_array(input_array=audio, 
        #                            sample_rate_in=self.sample_rate)
    
        # apply pitch shifting
        if np.random.rand() < 0.5:
            sgn = np.random.choice([-1,1])
            factor = sgn * np.random.rand() * 12.0     
            tfm = sox.Transformer()        
            tfm.pitch(factor)
            audio = tfm.build_array(input_array=audio.squeeze().numpy(), 
                                    sample_rate_in=self.sample_rate)
            audio = torch.from_numpy(audio.astype('float32')).view(1,-1)

        # apply a lowpass filter
        if np.random.rand() < 0.25:
            cutoff = (np.random.rand() * 4000) + 4000
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="lowpass", 
                                      fs=self.sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # apply a highpass filter
        if np.random.rand() < 0.25:
            cutoff = (np.random.rand() * 1000) + 20
            sos = scipy.signal.butter(2, 
                                      cutoff, 
                                      btype="highpass", 
                                      fs=self.sample_rate, 
                                      output='sos')
            audio_filtered = scipy.signal.sosfilt(sos, audio.numpy())
            audio = torch.from_numpy(audio_filtered.astype('float32'))

        # add white noise
        if np.random.rand() < 0.1:
            wn = (torch.rand(audio.shape) * 2) - 1
            g = 10**(-(np.random.rand() * 20) - 12)/20
            audio = audio + (g * wn)

        return audio, target