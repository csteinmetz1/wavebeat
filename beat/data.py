import os
import sys
import glob
import torch 
import julius
import torchaudio
import numpy as np

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
                 augment=True):
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
        """
        self.audio_dir = audio_dir
        self.annot_dir = annot_dir
        self.sample_rate = sample_rate
        self.length = length
        self.preload = preload
        self.half = half
        self.fraction = fraction
        self.augment = augment

        #if self.subset == "full":

        # for now we just load of all the data as training data
        self.audio_files = glob.glob(os.path.join(self.audio_dir, "**", "*.wav"))
        self.audio_files.sort() # sort the list of audio files

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

        if self.augment:
            if np.random.rand() > 0.6:      # random gain from 0dB to -12 dB
                audio = audio * (10**(-(np.random.rand() * 12)/20))   
            if np.random.rand() > 0.5:      # phase inversion
                audio = -audio                              
            if np.random.rand() > 0.2:      # apply compression
                audio = torch.tanh(audio)      
            if np.random.rand() > 0.05:     # drop frames
                zero_size = int(self.length*0.1)
                start = np.random.randint(audio.shape[-1] - zero_size - 1)
                stop = start + zero_size
                audio[:,start:stop] = 0             

        # now get the annotation information
        beat_samples, beat_indices = self.load_annot(self.annot_files[idx])

        # now we construct the target sequence with beat (1) and no beat (0)
        N = audio.shape[-1]
        target = torch.zeros(1,N)

        target[:,beat_samples] = 1

        # now we take a random crop of both
        if (N - self.length - 1) < 0:
            start = 0
            stop = self.length
        else:
            start = np.random.randint(N - self.length - 1)
            stop = start + self.length

        audio = audio[:,start:stop]
        target = target[:,start:stop]

        # check the length 
        if audio.shape[-1] < self.length:
            pad_size = self.length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (pad_size,0))
            target = torch.nn.functional.pad(target, (pad_size,0))

        return audio, target

    def load_annot(self, filename):

        with open(filename, 'r') as fp:
            lines = fp.readlines()
        
        beat_samples = [] # array of samples containing beats
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
            time_samples = int(float(time_sec) * (self.sample_rate))

            beat_samples.append(time_samples)
            beat_indices.append(beat_one_hot)

        return beat_samples, beat_indices