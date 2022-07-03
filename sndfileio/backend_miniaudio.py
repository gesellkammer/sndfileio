from __future__ import annotations
from .datastructs import SndInfo
from . import util
import numpy as np 
import miniaudio
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .datastructs import sample_t
    from typing import Iterator


def _readfragment(path: str, start: float, end: float
                  ) -> sample_t:
    info = miniaudio.mp3_get_file_info(path)
    seek_frame = int(start * info.sample_rate)
    if end == 0:
        frames_to_read = info.num_frames - seek_frame
    else:
        frames_to_read = int(end * info.sample_rate) - seek_frame
    buf = next(miniaudio.mp3_stream_file(path, frames_to_read=frames_to_read, seek_frame=seek_frame))
    samples = np.asarray(buf, dtype=float)
    samples /= 2**15
    return samples, info.sample_rate


def mp3read_chunked(path: str, chunksize: int, start=0., stop=0.
                    ) -> Iterator[np.ndarray]:
    info = miniaudio.mp3_get_file_info(path)
    sr = info.sample_rate
    seek_frame = int(start*sr)
    if stop == 0:
        frames_to_read = info.num_frames - seek_frame
    else:
        frames_to_read = min(info.num_frames, int(sr * stop)) - seek_frame
    assert frames_to_read > 0
    nchannels = info.nchannels
    for buf in miniaudio.mp3_stream_file(path, frames_to_read=chunksize, seek_frame=seek_frame):
        samples = np.asarray(buf, dtype=float)
        samples /= 2**15
        if nchannels > 1:
            samples.shape = (len(samples) // nchannels, nchannels)
        if frames_to_read < len(samples):
            samples = samples[:frames_to_read]
            yield samples
            return
        else:
            yield samples
            frames_to_read -= len(samples)


def mp3read(path: str, start=0., end=0.) -> sample_t:
    """
    Reads a mp3 files completely into an array
    """
    if start > 0 or end > 0:
        return _readfragment(path, start, end)
    decoded = miniaudio.mp3_read_file_f32(path)
    npsamples = np.frombuffer(decoded.samples, dtype='float32').astype(float)
    if decoded.nchannels > 1:
        npsamples.shape = (decoded.num_frames, decoded.nchannels)
    return npsamples, decoded.sample_rate


_encodingFromFormat = {
    'SIGNED16': 'pcm16'
}


def makeSndInfo(info: miniaudio.SoundFileInfo, metadata: dict = None) -> SndInfo:
    encoding = _encodingFromFormat.get(info.file_format.name, '')
    bitrate = metadata.pop('bitrate', None)
    return SndInfo(samplerate=info.sample_rate,
                   nframes=info.num_frames,
                   channels=info.nchannels,
                   encoding=encoding,
                   fileformat=info.file_format.name,
                   metadata=metadata,
                   bitrate=bitrate)


def mp3info(path: str) -> SndInfo:
    info = miniaudio.mp3_get_file_info(path)
    metadata = util.tinytagMetadata(path)
    return makeSndInfo(info, metadata=metadata)


def ogginfo(path: str) -> SndInfo:
    info = miniaudio.vorbis_get_file_info(path)
    metadata = util.tinytagMetadata(path)
    return makeSndInfo(info, metadata=metadata)