import io
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

def plot_activations(ref_beats, est_beats, est_sm, sample_rate):

    plt.figure(figsize=(5,3))

    plt.vlines(ref_beats, 1.5, 2.0, colors='b')
    plt.vlines(est_beats, 1, 1.5, colors='r')

    t = np.arange(len(est_sm))/sample_rate
    plt.plot(t, est_sm)
    plt.ylim([0, 2])

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    plt.close('all')

    return image