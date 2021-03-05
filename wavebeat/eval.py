import madmom
import mir_eval
import numpy as np
import scipy.signal

def find_beats(t, p, 
                smoothing=127, 
                threshold=0.5, 
                distance=None, 
                sample_rate=44100, 
                beat_type="beat",
                filter_type="none",
                peak_type="simple"):
    # 15, 2

    # t is ground truth beats
    # p is predicted beats 
    # 0 - no beat
    # 1 - beat

    N = p.shape[-1]

    if filter_type == "savgol":
        # apply smoothing with savgol filter
        p = scipy.signal.savgol_filter(p, smoothing, 2)   
    elif filter_type == "cheby":
        sos = scipy.signal.cheby1(10,
                                  1, 
                                  45, 
                                  btype='lowpass', 
                                  fs=sample_rate,
                                  output='sos')
        p = scipy.signal.sosfilt(sos, p)

    # normalize the smoothed signal between 0.0 and 1.0
    p /= np.max(np.abs(p))                                     

    if peak_type == "simple":
        # by default, we assume that the min distance between beats is fs/4
        # this allows for at max, 4 BPS, which corresponds to 240 BPM 
        # for downbeats, we assume max of 1 downBPS
        if beat_type == "beat":
            if distance is None:
                distance = sample_rate / 4
        elif beat_type == "downbeat":
            if distance is None:
                distance = sample_rate / 2
        else:
            raise RuntimeError(f"Invalid beat_type: `{beat_type}`.")

        # apply simple peak picking
        est_beats, heights = scipy.signal.find_peaks(p, height=threshold, distance=distance)

    elif peak_type == "cwt":
        # use wavelets
        est_beats = scipy.signal.find_peaks_cwt(p, np.arange(1,50))

    # compute the locations of ground truth beats
    ref_beats = np.squeeze(np.argwhere(t==1).astype('float32'))
    est_beats = est_beats.astype('float32')

    # compute beat points (samples) to seconds
    ref_beats /= float(sample_rate)
    est_beats /= float(sample_rate)

    # store the smoothed ODF
    est_sm = p

    return ref_beats, est_beats, est_sm

def evaluate(pred, target, target_sample_rate, use_dbn=False):

    t_beats = target[0,:]
    t_downbeats = target[1,:]
    p_beats = pred[0,:]
    p_downbeats = pred[1,:]

    ref_beats, est_beats, _ = find_beats(t_beats.numpy(), 
                                        p_beats.numpy(), 
                                        beat_type="beat",
                                        sample_rate=target_sample_rate)

    ref_downbeats, est_downbeats, _ = find_beats(t_downbeats.numpy(), 
                                                p_downbeats.numpy(), 
                                                beat_type="downbeat",
                                                sample_rate=target_sample_rate)

    if use_dbn:
        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55,
            max_bpm=215,
            transition_lambda=100,
            fps=target_sample_rate,
            online=False)

        downbeat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=10,
            max_bpm=75,
            transition_lambda=100,
            fps=target_sample_rate,
            online=False)

        beat_pred = pred[0,:].clamp(1e-8, 1-1e-8).view(-1).numpy()
        downbeat_pred = pred[1,:].clamp(1e-8, 1-1e-8).view(-1).numpy()

        est_beats = beat_dbn.process_offline(beat_pred)
        est_downbeats = downbeat_dbn.process_offline(downbeat_pred)

    # evaluate beats - trim beats before 5 seconds.
    ref_beats = mir_eval.beat.trim_beats(ref_beats)
    est_beats = mir_eval.beat.trim_beats(est_beats)
    beat_scores = mir_eval.beat.evaluate(ref_beats, est_beats)

    # evaluate downbeats - trim beats before 5 seconds.
    ref_downbeats = mir_eval.beat.trim_beats(ref_downbeats)
    est_downbeats = mir_eval.beat.trim_beats(est_downbeats)
    downbeat_scores = mir_eval.beat.evaluate(ref_downbeats, est_downbeats)

    return beat_scores, downbeat_scores