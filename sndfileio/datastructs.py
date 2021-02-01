from typing import NamedTuple
import numpy as np


class Sample(NamedTuple):
    samples: np.ndarray
    sr: int


class SndInfo:
    """
    A structure to hold information about a soundfile

    Attributes:
        samplerate: the samplerate of the soundfile
        nframes: the number of frames
        channels: the number of channels per frame
        encoding: a string describing the encoding of the data.
        fileformat: the format of the soundfile ("wav", "aif", "flac", "mp3")
            This is usually the same as the extension of the soundfile read
    """

    def __init__(self, samplerate:int, nframes:int, channels:int, 
                 encoding:str, fileformat:str
                 ) -> None:
        self.samplerate: int = samplerate
        self.nframes: int = nframes
        self.channels: int = channels
        self.encoding: str = encoding
        self.fileformat: str = fileformat

    @property
    def duration(self) -> float:
        """ The duration of the soundfile, in seconds """
        return self.nframes / self.samplerate

    def __repr__(self):
        return """------------------
samplerate : %d
nframes    : %d
channels   : %d
encoding   : %s
fileformat : %s
duration   : %.3f s""" % (
            self.samplerate,
            self.nframes,
            self.channels,
            self.encoding,
            self.fileformat,
            self.duration
        )
