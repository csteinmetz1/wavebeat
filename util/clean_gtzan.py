import os
import json
import glob
import numpy as np
import soundfile as sf

audio_dir = "/home/cjstein/datasets/gtzan"
annot_dir = "/home/cjstein/datasets/GTZAN-Rhythm/jams"

audio_files = glob.glob(os.path.join(audio_dir, "**", "*.au"))
annot_files = glob.glob(os.path.join(annot_dir, "*.jams"))

#for idx, audio_file in enumerate(audio_files):
#    audio, sr = sf.read(audio_file)
#    print(idx, audio.shape, sr)

#    filename = audio_file.replace(".au", ".wav")
    #sf.write(filename, audio, sr)

for idx, annot_file in enumerate(annot_files):

    with open(annot_file, 'r') as fp:
        jam = json.load(fp)

    beats = []
    downbeats = []
    for val in jam['annotations']:
        if 'annotation_type' not in val['sandbox']:
            continue
        if val['sandbox']['annotation_type'] == "beat":
            beats_data = val['data']
            for beat in beats_data:
                beats.append(beat['time'])
        elif val['sandbox']['annotation_type'] == "downbeat":
            beats_data = val['data']
            for beat in beats_data:
                downbeats.append(beat['time'])
    
    if len(beats) == 0 or len(downbeats) == 0:
        print(f"Found {len(beats)} and {len(downbeats)}, skipping {annot_file}")
        continue

    # now write out a new file to disk with common format
    new_annot_file = annot_file.replace(".jams", ".txt")

    # convert out lists to arrays
    beats = np.array(beats)
    downbeats = np.array(downbeats)

    first_downbeat_idx = np.nonzero(beats == downbeats[0])[0][0]
    beat_idx = 1

    with open(new_annot_file, 'w') as fp:
        for beat in beats:
            if beat in downbeats:
                beat_idx = 1
            else:
                beat_idx += 1
            line = f"{beat} {beat_idx}"
            fp.write(f"{beat} {beat_idx}")

    #break

