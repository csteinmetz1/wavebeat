import io
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
                     est_downbeats_sm=None):

    plt.figure(figsize=(12,3))

    plt.vlines(ref_beats, 1.55, 1.95, colors='lightcoral')
    plt.vlines(est_beats, 1.05, 1.45, colors='lightsteelblue') 

    if ref_downbeats is not None:
        plt.vlines(ref_downbeats, 1.75, 1.95, colors='red')
    if est_downbeats is not None:
        plt.vlines(est_downbeats, 1.25, 1.45, colors='blue')

    t = np.arange(len(est_sm))/sample_rate
    plt.plot(t, 0.45 * est_sm, c="lightsteelblue")

    if est_downbeats_sm is not None:
        plt.plot(t, (0.45 * est_downbeats_sm) + 0.5, c="blue")

    plt.ylim([0, 2])
    if len(ref_beats > 0):
        plt.xlim([ref_beats[0], ref_beats[-1]])

    plt.xlabel("Time (s)")
    plt.yticks([0.25, 0.75, 1.25, 1.75], ['Beat', 'Downbeat', 'Pred.', 'Target'])

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    plt.close('all')

    return image