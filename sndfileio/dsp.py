"""
Helper functions for doing simple dsp filtering
"""

from __future__ import annotations
import numpy as np
import warnings
from typing import Tuple, Sequence as Seq, Union
from .util import apply_multichannel


def lowpass_cheby(samples:np.ndarray, freq:float, sr:int, maxorder=12) -> np.ndarray:
    """
    Cheby2-Lowpass Filter

    Filter data by passing data only below a certain frequency.
    The main purpose of this cheby2 filter is downsampling.
    
    This method will iteratively design a filter, whose pass
    band frequency is determined dynamically, such that the
    values above the stop band frequency are lower than -96dB.

    Args:
        samples: Data to filter, type numpy.ndarray.
        freq : The frequency above which signals are attenuated with 95 dB
        sr: Sampling rate in Hz.
        maxorder: Maximal order of the designed cheby2 filter

    Returns:
        the filtered array
    """
    if freq > sr*0.5:
        raise ValueError("Can't filter freq. above nyquist")
    b, a, freq_passband = lowpass_cheby2_coeffs(freq, sr, maxorder)
    from scipy import signal
    return signal.lfilter(b, a, samples)


# noinspection PyTupleAssignmentBalance
def lowpass_cheby2_coeffs(freq:float, sr:int, maxorder=12
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Args:
        freq: The frequency above which signals are attenuated with 95 dB
        sr: Sampling rate in Hz.
        maxorder: Maximal order of the designed cheby2 filter

    Returns:
         a tuple (b coeffs, a coeffs, freq_passband)
    """
    from scipy import signal
    nyquist = sr * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freq / nyquist  # stop band frequency
    wp = ws              # pass band frequency
    # raise for some bad scenarios
    if ws > 1:
        ws = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    while True:
        if order <= maxorder:
            break
        wp = wp * 0.99
        order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=False)
    b, a = signal.cheby2(order, rs, wn, btype='low', analog=False, output='ba')
    return b, a, wp*nyquist


def filter_butter_coeffs(filtertype:str,
                         freq: Union[float, Tuple[float, float]],
                         sr:int,
                         order=5
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    calculates the coefficients for a digital butterworth filter

    Args:
        filtertype: 'low', 'high', 'band'
        freq: cutoff freq. In the case of 'band': (low, high)
        sr: the sample rate of the data
        order: the order of the filter

    Returns:
         a tuple (b, a) with the corresponding coefficients

    """
    assert filtertype in ('low', 'high', 'band')
    from scipy import signal
    nyq = 0.5*sr
    if isinstance(freq, tuple):
        assert filtertype == 'band'
        low, high = freq
        low  /= nyq
        high /= nyq
        b, a = signal.butter(order, [low, high], btype='band', output='ba')
    else:
        freq = freq / nyq
        b, a = signal.butter(order, freq, btype=filtertype, output='ba')
    return b, a


def filter_butter(samples: np.ndarray, sr:int, filtertype:str, freq:float, order=5
                  ) -> np.ndarray:
    """
    Filters the samples with a digital butterworth filter

    Args:
        samples: mono samples
        filtertype: 'low', 'band', 'high'
        freq : for low or high, the cutoff freq; for band, (low, high)
        sr: the sampling-rate
        order: the order of the butterworth filter

    Returns:
         the filtered samples

    .. note::
    
        calls filter_butter_coeffs to calculate the coefficients
    """
    from scipy import signal
    assert filtertype in ('low', 'high', 'band')
    b, a = filter_butter_coeffs(filtertype, freq, sr, order=order)
    return apply_multichannel(samples, lambda data:signal.lfilter(b, a, data))
    

def filter_butter_plot_freqresponse(b:Seq[float], a:Seq[float],
                                    samplerate:int, numpoints=2000) -> None:
    """
    Plot the freq. response of the digital butterw. filter 
    defined by the coeffs. (b, a) at `samplerate`

    .. seealso:: filter_butter_coeffs

    """
    from scipy import signal
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is needed for plotting")
    plt.figure(1)
    plt.clf()
    w, h = signal.freqz(b, a, worN=numpoints)
    plt.semilogx((samplerate*0.5/np.pi) * w, np.abs(h))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()
    