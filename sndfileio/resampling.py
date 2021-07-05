from __future__ import annotations
import scipy.signal as sig
import numpy as np
from .dsp import lowpass_cheby
import logging
from math import gcd
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional as Opt, Callable


class BackendNotAvailable(Exception):
    pass


logger = logging.getLogger("sndfileio")


def _applyMultichan(samples: np.ndarray,
                    func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Apply func to each channel of audio data in samples
    """
    if len(samples.shape) == 1 or samples.shape[1] == 1:
        newsamples = func(samples)
    else:
        y = np.array([])
        for i in range(samples.shape[1]):
            y = np.concatenate((y, func(samples[:,i])))
        newsamples = y.reshape(samples.shape[1], -1).T
    return newsamples    
    

def _resample_scipy(samples: np.ndarray, sr:int, newsr:int, window='hanning'
                    ) -> np.ndarray:
    try:
        from scipy.signal import resample
    except ImportError:
        raise BackendNotAvailable()

    ratio = newsr/sr
    lenNewSamples = int(ratio * len(samples) + 0.5)

    return _applyMultichan(samples, 
                           lambda S: resample(S, lenNewSamples, window=window))


def _resample_samplerate(samples:np.ndarray, sr:int, newsr:int) -> np.ndarray:
    """
    Uses https://github.com/tuxu/python-samplerate
    """
    try:
        from samplerate import resample
    except ImportError:
        raise BackendNotAvailable()

    ratio = newsr/sr
    return _applyMultichan(samples,
                           lambda S: resample(S, ratio, 'sinc_best'))

#######################################################

# global cache of resamplers
_precomputed_filters = {}


def _nnresample_compute_filt(up, down, beta=5.0, L=32001):
    r"""
    Computes a filter to resample a signal from rate "down" to rate "up"
    
    Parameters
    ----------
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    beta : float
        Beta factor for Kaiser window.  Determines tradeoff between
        stopband attenuation and transition band width
    L : int
        FIR filter order.  Determines stopband attenuation.  The higher
        the better, ath the cost of complexity.
        
    Returns
    -------
    filt : array
        The FIR filter coefficients
        
    Notes
    -----
    This function is to be used if you want to manage your own filters
    to be used with scipy.signal.resample_poly (use the `window=...`
    parameter).  WARNING: Some versions (at least 0.19.1) of scipy
    modify the passed filter, so make sure to make a copy beforehand:
    
    out = scipy.signal.resample_poly(in up, down, window=numpy.array(filt))
    """
    
    # Determine our up and down factors
    g = gcd(up, down)
    up = up//g
    down = down//g
    max_rate = max(up, down)

    sfact = np.sqrt(1+(beta/np.pi)**2)
            
    # generate first filter attempt: with 6dB attenuation at f_c.
    init_filt = sig.fir_filter_design.firwin(L, 1/max_rate, window=('kaiser', beta))
    
    # convert into frequency domain
    N_FFT = 2**19
    NBINS = N_FFT/2+1
    paddedfilt = np.zeros(N_FFT)
    paddedfilt[:L] = init_filt
    ffilt = np.fft.rfft(paddedfilt)
    
    # now find the minimum between f_c and f_c+sqrt(1+(beta/pi)^2)/L
    bot = int(np.floor(NBINS/max_rate))
    top = int(np.ceil(NBINS*(1/max_rate + 2*sfact/L)))
    firstnull = (np.argmin(np.abs(ffilt[bot:top])) + bot)/NBINS
    
    # generate the proper shifted filter
    return sig.fir_filter_design.firwin(L, -firstnull+2/max_rate, window=('kaiser', beta))


def _resample_nnresample(samples: np.ndarray, sr:int, newsr:int) -> np.ndarray:
    return _applyMultichan(samples,
                           lambda S: _resample_nnresample2(S, newsr, sr)[:-1])


def _resample_nnresample_package(samples: np.ndarray, sr:int, newsr:int) -> np.ndarray:
    return _applyMultichan(samples,
                           lambda S: _resample_nnresample_package_mono(S, newsr, sr)[:-1])


def _resample_nnresample_package_mono(s:np.ndarray, up:int, down:int, **kws) -> np.ndarray:
    import nnresample
    return nnresample.resample(s, up, down, axis=0, fc='nn', **kws)


def _resample_nnresample2(s:np.ndarray, up:int, down:int, beta=5.0, L=16001, axis=0
                          ) -> np.ndarray:
    """
    Taken from https://github.com/jthiem/nnresample

    Resample a signal from rate "down" to rate "up"

    Args:
        s (array): The data to be resampled.
        up (int): The upsampling factor.
        down (int): The downsampling factor.
        beta (float): Beta factor for Kaiser window. Determines tradeoff between
            stopband attenuation and transition band width
        L (int): FIR filter order.  Determines stopband attenuation.  The higher
            the better, ath the cost of complexity.
        axis (int): int, optional. The axis of `s` that is resampled. Default is 0.
        
    Returns:
        The resampled array.

    .. note::
    
        The function keeps a global cache of filters, since they are
        determined entirely by up, down, beta, and L.  If a filter
        has previously been used it is looked up instead of being
        recomputed.
    """
    # check if a resampling filter with the chosen parameters already exists
    params = (up, down, beta, L)
    if params in _precomputed_filters.keys():
        # if so, use it.
        filt = _precomputed_filters[params]
    else:
        # if not, generate filter, store it, use it
        filt = _nnresample_compute_filt(up, down, beta, L)
        _precomputed_filters[params] = filt
    return sig.resample_poly(s, up, down, window=np.array(filt), axis=axis)


def _resample_obspy(samples:np.ndarray, sr:int, newsr:int, window='hanning', lowpass=True
                    ) -> np.ndarray:
    """
    Resample using Fourier method. The same as resample_scipy but with
    low-pass filtering for upsampling
    """
    from scipy.signal import resample
    from math import ceil
    factor = sr/float(newsr)
    if newsr < sr and lowpass:
        # be sure filter still behaves good
        if factor > 16:
            logger.info("Automatic filter design is unstable for resampling "
                        "factors (current sampling rate/new sampling rate) " 
                        "above 16. Manual resampling is necessary.")
        freq = min(sr, newsr) * 0.5 / float(factor)
        logger.debug(f"resample_obspy: lowpass {freq}")
        samples = lowpass_cheby(samples, freq=freq, sr=sr, maxorder=12)
    num = int(ceil(len(samples) / factor))

    return _applyMultichan(samples, 
                           lambda S: resample(S, num, window=window))


def resample(samples: np.ndarray, oldsr:int, newsr:int) -> np.ndarray:
    """
    Resample `samples` with given samplerate `sr` to new samplerate `newsr`

    Args:
        samples: mono or multichannel frames
        oldsr: original samplerate
        newsr: new sample rate

    Returns:
        the new samples
    """
    backends = [
        _resample_samplerate,   # turns the samples into float32, which is ok for audio
        _resample_nnresample_package,  # nnresample packaged version
        _resample_nnresample,   # (builtin) very good results, follows libsamplerate closely
        _resample_obspy,        # these last two introduce some error at the first samples
        _resample_scipy
    ]

    for backend in backends:
        try:
            return backend(samples, oldsr, newsr)
        except BackendNotAvailable:
            pass
