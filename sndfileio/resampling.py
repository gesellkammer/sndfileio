import warnings
import numpy as np
from .dsp import lowpass_cheby2
from typing import List, Optional as Opt, Callable

def _apply_by_channel(samples, func):
    # type: (np.ndarray, Callable[[np.ndarray], np.ndarray]) -> np.ndarray
    """
    apply func to each channel of audio data in samples
    """
    if len(samples.shape) == 1 or samples.shape[1] == 1:
        newsamples = func(samples)
    else:
        y = np.array([])
        for i in range(samples.shape[1]):
            y = np.concatenate((y, func(samples[:,i])))
        newsamples = y.reshape(samples.shape[1], -1).T
    return newsamples    
    

def _resample_scipy(samples, sr, newsr, window='hanning'):  
    # type: (np.ndarray, int, int, str) -> np.ndarray
    try:
        import scipy.signal
    except ImportError:
        return None

    ratio = newsr/sr
    num_new_samples = int(ratio * len(samples) + 0.5)
    def func(samples):
        return scipy.signal.resample(samples, num_new_samples, window=window)
    return _apply_by_channel(samples, func)


def _resample_scikits(samples, sr, newsr):
    # type: (np.ndarray, int, int) -> np.ndarray
    try:
        import scikits.samplerate
    except ImportError:
        return None
    ratio = newsr / sr
    return scikits.samplerate.resample(samples, ratio, 'sinc_best')

def _resample_obspy(samples, sr, newsr, window='hanning', lowpass=True):
    # type: (np.ndarray, int, int, str, bool) -> np.ndarray
    """
    Resample using Fourier method.
    """
    import scipy.signal    
    factor = sr/float(newsr)
    if lowpass:
        # be sure filter still behaves good
        if factor > 16:
            msg = "Automatic filter design is unstable for resampling " + \
                  "factors (current sampling rate/new sampling rate) " + \
                  "above 16. Manual resampling is necessary."
            warnings.warn(msg)
        freq = min(sr, newsr) * 0.5 / float(factor)
        samples = lowpass_cheby2(samples, freq=freq, sr=sr, maxorder=12)
    num = len(samples) / factor
    def func(samples):
        return scipy.signal.resample(samples, num, window=window)
    return _apply_by_channel(samples, func)

def resample(samples, oldsr, newsr):
    # type: (np.ndarray, int, int) -> np.ndarray
    """
    Resample `samples` with given samplerate `sr` to new samplerate `newsr`

    samples: mono or multichannel frames
    oldsr  : original samplerate
    newsr  : new sample rate

    Returns --> the new samples
    """
    backends = [_resample_scikits, _resample_obspy, _resample_scipy] # type: List[Callable[[np.ndarray, int, int], Opt[np.ndarray]]]
    for backend in backends:
        newsamples = backend(samples, oldsr, newsr)
        if newsamples is not None:
            return newsamples
    
    raise ImportError(
        "no backends present to perform resampling."
        "At least scikits.samplerate or scipy should be present"
        )