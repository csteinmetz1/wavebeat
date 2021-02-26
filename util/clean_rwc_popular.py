import os
import glob
import soundfile as sf

audio_dir = "/home/cjstein/datasets/rwc_popular/audio"
annot_dir = "/home/cjstein/datasets/rwc_popular/beat"

audio_files = glob.glob(os.path.join(audio_dir, "**", "*.aiff"))
#annot_files = glob.glob(os.path.join(annot_dir, "**", "*.txt"))

for idx, audio_file in enumerate(audio_files):
    audio, sr = sf.read(audio_file)
    print(idx, audio.shape)

    audioL = audio[:,0]
    audioR = audio[:,1]
    audioM = (audioL + audioR)/2

    left_filename = audio_file.replace(".aiff", "_L.wav")
    right_filename = audio_file.replace(".aiff", "_R.wav")
    mono_filename = audio_file.replace(".aiff", "_L+R.wav")

    sf.write(left_filename, audioL, sr)
    sf.write(right_filename, audioR, sr)
    sf.write(mono_filename, audioM, sr)

