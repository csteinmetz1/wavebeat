import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

def plot_activations(input, target):

    plt.figure(figsize=(5,3))
    plt.plot(input, label='Prediction')
    plt.plot(target, label='Target')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)

    image = PIL.Image.open(buf)
    image = ToTensor()(image)

    plt.close('all')

    return image