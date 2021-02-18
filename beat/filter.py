import torch
import scipy.signal

class FIRFilter(torch.nn.Module):
    """ FIR filtering module.
    Args:
        filter_type (str): Shape of the desired FIR filter ("lp"). Default: "lp"
        fc (float): Cutoff frequency in Hz. 
        ntaps (int): Number of FIR filter taps for constructing FIR filter. Default: 127
        plot (bool): Plot the magnitude respond of the filter. Default: False
    """

    def __init__(self, filter_type="lp", fc=1000, fs=44100, ntaps=127, plot=False):
        """Initilize FIR filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.fc = fc 
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "lp":
            # we fit to N tap FIR filter with window method
            taps = scipy.signal.firwin(ntaps, fc, fs=fs, pass_zero=False)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=ntaps, bias=False, padding=ntaps//2)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype('float32')).view(1,1,-1)

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        input = torch.nn.functional.conv1d(input, self.fir.weight.data, padding=self.ntaps//2)
        target = torch.nn.functional.conv1d(target, self.fir.weight.data, padding=self.ntaps//2)
        return input, target