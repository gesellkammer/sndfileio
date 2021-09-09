from .datastructs import SndInfo
import pydub
import time
import numpy as np
from typing import Tuple


_cache = {}


def _audiosegment(path:str, timeout=10) -> pydub.AudioSegment:
    """
    Args:
        timeout: the time (in seconds) this audiosegment is cached
    """
    if path in _cache:
        audiosegment, t = _cache[path]
        if time.time() - t < timeout:
            return audiosegment
    audiosegment = pydub.AudioSegment.from_mp3(path)
    _cache[path] = (audiosegment, time.time())
    return audiosegment


def mp3info(path:str) -> SndInfo:
    f = _audiosegment(path)
    return SndInfo(samplerate=f.frame_rate, nframes=f.frame_count(), 
                   channels=f.channels, encoding='pcm16', fileformat='mp3')


def mp3read(path:str) -> Tuple[np.ndarray, int]:
    """
    Returns a tuple (samples, samplerate) where samples 
    is a numpy array (float, between -1 and 1)
    """
    f = _audiosegment(path)
    samples = f.get_array_of_samples()
    samplesnp = np.frombuffer(samples, np.int16, len(samples)) / (2**15)
    if f.channels > 1:
        samplesnp.shape = (int(f.frame_count()), f.channels)
    return samplesnp, f.frame_rate
