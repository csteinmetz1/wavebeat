import torch

class GlobalMSELoss(torch.nn.Module):
    def __init__(self):
        super(GlobalMSELoss, self).__init__()

    def forward(self, input, target):
        
        # beat errors
        target_beats = target[target == 1]
        input_beats = input[target == 1]

        beat_loss = torch.nn.functional.mse_loss(input_beats, target_beats)

        # no beat errors
        target_no_beats = target[target == 0]
        input_no_beats = input[target == 0]

        no_beat_loss = torch.nn.functional.mse_loss(input_beats, target_beats)

        return no_beat_loss + beat_loss, beat_loss, no_beat_loss