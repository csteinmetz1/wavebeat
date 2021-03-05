import io
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

def plot_activations(ref_beats, 
                     est_beats, 
                     est_sm, 
                     sample_rate, 
                     ref_downbeats=None, 
                     est_downbeats=None, 
                     est_downbeats_sm=None,
                     song_name=None):

    plt.figure(figsize=(12,3))

    plt.vlines(ref_beats, 1.55, 1.75, colors='lightcoral')
    plt.vlines(est_beats, 1.05, 1.25, colors='lightsteelblue') 

    if ref_downbeats is not None:
        plt.vlines(ref_downbeats, 1.80, 1.95, colors='red')
    if est_downbeats is not None:
        plt.vlines(est_downbeats, 1.30, 1.45, colors='blue')

    t = np.arange(len(est_sm))/sample_rate
    plt.plot(t, 0.45 * est_sm, c="lightsteelblue")

    if est_downbeats_sm is not None:
        plt.plot(t, (0.45 * est_downbeats_sm) + 0.5, c="blue")

    plt.ylim([0, 2])
    if len(ref_beats) > 0:
        plt.xlim([ref_beats[0], ref_beats[-1]])

    if song_name is not None:
        plt.title(f"{song_name}")
    plt.xlabel("Time (s)")
    plt.yticks([0.25, 0.75, 1.25, 1.75], ['Beat', 'Downbeat', 'Pred.', 'Target'])
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    plt.close('all')

    return image

def plot_histogram(songs):

    beat_f1_scores = []
    downbeat_f1_scores = []

    for song in songs:
        beat_f1_scores.append(song["Beat F-measure"])
        downbeat_f1_scores.append(song["Downbeat F-measure"])

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,6))

    axs[0].hist(beat_f1_scores, bins=10)
    axs[0].set_xlabel('Beat F-measure')

    axs[1].hist(downbeat_f1_scores, bins=10)
    axs[1].set_xlabel('Downbeat F-measure')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    plt.close('all')

    return image

def make_table(songs, sort_key="Beat F-measure"):

    # first sort by ascending f-measure on beats
    songs = sorted(songs, key=lambda k: k[sort_key])

    table = ""
    table += "| File     | Genre | Time Sig.| Beat F-measure |  Downbeat F-measure |\n"
    table += "|:---------|-------|----------|---------------:|--------------------:|\n"

    for song in songs:
        table += f"""| {os.path.basename(song["Filename"])} |"""
        table += f"""  {song["Genre"]} |"""
        table += f"""  {song["Time signature"]} |"""
        table += f"""{song["Beat F-measure"]:0.3f} | """
        table += f"""{song["Downbeat F-measure"]:0.3f} |\n"""

    return table