from __future__ import annotations
import numpy as np
from .datastructs import SndWriter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import lameenc


def convertSamplesToInt16(y: np.ndarray) -> np.ndarray:
    """Convert floating-point numpy array of audio samples to int16."""
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")
    return (y*np.iinfo(np.int16).max).astype(np.int16)


def _create_encoder(sr: int, numchannels: int, bitrate: int, quality: int
                    ) -> lameenc.Encoder:
    import lameenc
    encoder = lameenc.Encoder()
    encoder.silence()
    encoder.set_channels(numchannels)
    encoder.set_quality(quality)
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sr)
    return encoder


class LameencWriter(SndWriter):
    def __init__(self, outfile: str, sr: int, bitrate: int, quality: int = 3):
        super().__init__(outfile=outfile, sr=sr, fileformat='mp3', encoding='pcm16')
        self._encoder = _create_encoder(sr=self.sr, numchannels=2, bitrate=bitrate,
                                        quality=quality)

    def write(self, frames: np.ndarray) -> None:
        if not self._file:
            numchannels = frames.shape[1] if len(frames.shape) == 2 else 1
            self._encoder.set_channels(numchannels)
            self._file = open(self.outfile, "wb")
        databytes = convertSamplesToInt16(frames).tobytes()
        mp3data = self._encoder.encode(databytes)
        self._file.write(mp3data)

    def close(self):
        if self._file is None:
            raise IOError("Can't close, since this file was never open")
        mp3data = self._encoder.flush()
        self._file.write(mp3data)
        self._file.close()
        self._file = None

