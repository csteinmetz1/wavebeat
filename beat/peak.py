import numpy as np
import scipy.signal

def find_beats(t, p, smoothing=15, threshold=0.8, distance=None, sample_rate=44100):

    # t is ground truth beats
    # p is predicted beats 
    # 0 - no beat
    # 1 - beat

    N = p.shape[-1]
    p = scipy.signal.savgol_filter(p, smoothing, 2)    # apply smoothing with savgol filter
    p /= np.max(p)                                     # normalize the smoothed signal between 0.0 and 1.0

    # by default, we assume that the min distance between beats is fs/4
    # this allows for at max, 4 BPS, which corresponds to 240 BPM 
    if distance is None:
        distance = sample_rate / 4

    # perform peak picking given the supplied parameters
    est_beats, heights = scipy.signal.find_peaks(p, height=threshold, distance=distance)

    # compute the locations of ground truth beats
    ref_beats = np.squeeze(np.argwhere(t==1).astype('float32'))
    est_beats = est_beats.astype('float32')

    # compute beat points (samples) to seconds
    ref_beats /= float(sample_rate)
    est_beats /= float(sample_rate)

    # store the smoothed ODF
    est_sm = p

    return ref_beats, est_beats, est_sm