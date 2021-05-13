from typing import NamedTuple, Dict, Any
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
        metadata: any metadata set in the file. Metadata, if present, is
            presented in the form of a dict[str, str], where keys are restricted
            to a subset of possible values: 'comment', 'title', 'artist',
            'album', 'tracknumber'
    """

    def __init__(self, samplerate:int, nframes:int, channels:int, 
                 encoding:str, fileformat:str, metadata: Dict[str, str]=None,
                 extrainfo: Dict[str, Any] = None
                 ) -> None:
        self.samplerate: int = samplerate
        self.nframes: int = nframes
        self.channels: int = channels
        self.encoding: str = encoding
        self.fileformat: str = fileformat
        self.metadata = metadata
        self.extrainfo = extrainfo

    @property
    def duration(self) -> float:
        """ The duration of the soundfile, in seconds """
        return self.nframes / self.samplerate

    def __repr__(self):
        s = f"""samplerate : {self.samplerate}
nframes    : {self.nframes}
channels   : {self.channels}
encoding   : {self.encoding}
fileformat : {self.fileformat}
duration   : {self.duration:.3f}"""
        if self.metadata:
            s += f"\nmetadata   : {self.metadata}"
        return s
