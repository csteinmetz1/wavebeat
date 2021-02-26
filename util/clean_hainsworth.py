import os
import glob
import numpy as np
import soundfile as sf

data_filename = '/home/cjstein/datasets/hainsworth/data.txt'

# make a new directory to store associated beats for each wav
root_dir = os.path.dirname(data_filename)
beat_dir = os.path.join(root_dir, "beat")
wavs_dir = os.path.join(root_dir, "wavs")
if not os.path.isdir(beat_dir):
    os.makedirs(beat_dir)

with open(data_filename, 'r') as fp:
    lines = fp.readlines()

for idx, line in enumerate(lines):
    if idx < 13: continue
    items = line.split('<sep>')
    items = [item.strip('\t') for item in items]
    items = [item.strip(' ') for item in items]
    items = [item.strip(' \n') for item in items]

    # wave filename
    wav_filename = items[0]
    print(wav_filename)

    # 5 is the genre
    genre = items[5].replace('/', '_')

    # 10 are beat indices
    dirty_beat_samples = items[10].split(',')
    beat_samples = []
    for beat_sample in dirty_beat_samples:
        if "." in beat_sample:
            beat_samples.append(int(float(beat_sample)))
        else:
            beat_samples.append(int(beat_sample))
    beat_samples = np.array(beat_samples)
    print(len(beat_samples))

    # 11 are indicies within beats for downbeats
    downbeat_indices = [int(i)-1 for i in items[11].split(',')] 
    downbeat_indices = [db for db in downbeat_indices if db < len(beat_samples)]
    # this assume index is 1 in file
    downbeat_samples = beat_samples[downbeat_indices]

    # now load the audio file so we can check the sample rate
    wav_path = os.path.join(wavs_dir, wav_filename)
    data, sr = sf.read(wav_path)

    # convert beats and downbeats to seconds
    beat_sec = beat_samples / float(sr) 
    downbeat_sec = downbeat_samples / float(sr) 

    # make a genre directory for this in wavs and beats
    genre_beat_dir = os.path.join(beat_dir, genre)
    genre_wavs_dir = os.path.join(wavs_dir, genre)
    if not os.path.isdir(genre_beat_dir):
        os.makedirs(genre_beat_dir)
    if not os.path.isdir(genre_wavs_dir):
        os.makedirs(genre_wavs_dir)

    # move the wav tile to this new location
    new_wav_path = os.path.join(genre_wavs_dir, wav_filename)
    #print(new_wav_path)
    os.rename(wav_path, new_wav_path)

    # write out a beat file with
    # seconds   beat_type
    beat_filename = wav_filename.replace(".wav", ".txt")
    beat_path = os.path.join(genre_beat_dir, beat_filename)

    with open(beat_path, 'w') as fp:
        d = 0
        for idx, b in enumerate(beat_sec):
            if b in downbeat_sec:
                d = 1
            else:
                d += 1
            line = f"{b} {d}\n"
            fp.write(line)
