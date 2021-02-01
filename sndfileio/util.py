import numpy as np
from typing import Callable


def numchannels(samples:np.ndarray) -> int:
    """
    return the number of channels present in samples

    Args:
        samples: a numpy array as returned by sndread

    Returns:
        the number of channels in `samples`

    For multichannel audio, samples is always interleaved,
    meaning that samples[n] returns always a frame, which
    is either a single scalar for mono audio, or an array
    for multichannel audio.
    """
    if len(samples.shape) == 1:
        return 1
    else:
        return samples.shape[1]


def apply_multichannel(data:np.ndarray, func:Callable[[np.ndarray], np.ndarray]
                       ) -> np.ndarray:
    """
    Apply a function ``(samples1D) -> samples1D`` along the channels of a multichannel
    sample array, returning a multichannel sample array of the same shape

    Args:
        data: the samples
        func: the function to apply to each channel

    Returns:
        the resulting samples

    Example::

        >>> import numpy as np
        >>> source = np.array([[0.,  0.],
        ...                    [0.1, 0.2],
        ...                    [0.3, 0.4.],
        ...                    [0.5, 0.6]])
        >>> apply_multichannel(source, lambda chan: chan*0.5)
        array([[0.,   0. ]
               [0.05, 0.1]
               [0.15, 0.2]
               [0.25, 0.3]])
    """
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