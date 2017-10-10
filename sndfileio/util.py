import numpy as np
from typing import Callable


def numchannels(samples:np.ndarray) -> int:
    """
    return the number of channels present in samples

    samples: a numpy array as returned by sndread

    for multichannel audio, samples is always interleaved,
    meaning that samples[n] returns always a frame, which
    is either a single scalar for mono audio, or an array
    for multichannel audio.
    """
    if len(samples.shape) == 1:
        return 1
    else:
        return samples.shape[1]


def apply_multichannel(data:np.ndarray, func:Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    ch = numchannels(data)
    if ch == 1:
        return func(data)
    else:
        chans = []
        for i in range(ch):
            chans.append(func(data[:,i]))
        out = np.concatenate(chans)
        out.shape = (ch, len(data))
        return out.T


del Callable