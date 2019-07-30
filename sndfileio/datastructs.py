from typing import NamedTuple
import numpy as np


class Sample(NamedTuple):
    samples: np.ndarray
    sr: int


class SndInfo:

    def __init__(self, samplerate:int, nframes:int, channels:int, 
                 encoding:str, fileformat:str
                 ) -> None:
        self.samplerate = samplerate 
        self.nframes = nframes
        self.channels = channels
        self.encoding = encoding
        self.fileformat = fileformat

    @property
    def duration(self) -> float:
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
