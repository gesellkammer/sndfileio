from .datastructs import SndInfo, Sample
import numpy as np 
import miniaudio


def mp3read(path: str) -> np.ndarray:
    """
    Reads a mp3 files completely into an array
    """
    decoded = miniaudio.mp3_read_file_f32(path)
    npsamples = np.frombuffer(decoded.samples, dtype='float32').astype(float)
    if decoded.nchannels > 1:
        npsamples.shape = (decoded.num_frames, decoded.nchannels)
    return Sample(npsamples, decoded.sample_rate)


def mp3info(path: str) -> SndInfo:
    info = miniaudio.mp3_get_file_info(path)
    return SndInfo(samplerate=info.sample_rate,
                   nframes=info.num_frames,
                   channels=info.nchannels,
                   encoding='pcm16',
                   fileformat='mp3')
