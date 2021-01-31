from .datastructs import SndInfo, Sample
from typing import Iterator
import numpy as np 
import miniaudio


def _readfragment(path:str, start:float, end:float) -> Sample:
    info = miniaudio.mp3_get_file_info(path)
    seek_frame = int(start * info.sample_rate)
    if end == 0:
        frames_to_read = info.num_frames - seek_frame
    else:
        frames_to_read = int(end * info.sample_rate) - seek_frame
    buf = next(miniaudio.mp3_stream_file(path, frames_to_read=frames_to_read, seek_frame=seek_frame))
    samples = np.asarray(buf, dtype=float)
    samples /= 2**15
    return Sample(samples, info.sample_rate)


def mp3read_chunked(path:str, chunksize:int, skiptime:float=0.) -> Iterator[np.ndarray]:
    info = miniaudio.mp3_get_file_info(path)
    seek_frame = int(skiptime*info.sample_rate)
    for buf in miniaudio.mp3_stream_file(path, frames_to_read=chunksize, seek_frame=seek_frame):
        samples = np.asarray(buf, dtype=float)
        samples /= 2**15
        yield samples


def mp3read(path: str, start=0., end=0.) -> Sample:
    """
    Reads a mp3 files completely into an array
    """
    if start > 0 or end > 0:
        return _readfragment(path, start, end)
    decoded = miniaudio.mp3_read_file_f32(path)
    npsamples = np.frombuffer(decoded.samples, dtype='float32').astype(float)
    if decoded.nchannels>1:
        npsamples.shape = (decoded.num_frames, decoded.nchannels)
    return Sample(npsamples, decoded.sample_rate)


def mp3info(path: str) -> SndInfo:
    info = miniaudio.mp3_get_file_info(path)
    return SndInfo(samplerate=info.sample_rate,
                   nframes=info.num_frames,
                   channels=info.nchannels,
                   encoding='pcm16',
                   fileformat='mp3')
