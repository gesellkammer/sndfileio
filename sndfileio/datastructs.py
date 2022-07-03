from __future__ import annotations
import numpy as np
from . import util
import os

from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from typing import List, Optional as Opt, Dict, Any
    sample_t = Tuple[np.ndarray, int]


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
        bitrate: compression bitrate, only present when reading compressed files (mp3, ogg)
    """

    def __init__(self, samplerate: int, nframes: int, channels: int,
                 encoding: str, fileformat: str, metadata: Dict[str, str] = None,
                 extrainfo: Dict[str, Any] = None,
                 bitrate: int = None
                 ) -> None:
        self.samplerate: int = samplerate
        self.nframes: int = nframes
        self.channels: int = channels
        self.encoding: str = encoding
        self.fileformat: str = fileformat
        self.metadata = metadata
        self.extrainfo = extrainfo
        self.bitrate = bitrate

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
        if self.bitrate:
            s += f"\nbitrate    : {self.bitrate}"
        if self.metadata:
            s += f"\nmetadata   : {self.metadata}"
        return s


class SndWriter:
    """
    Class returned by :meth:`sndwrite_chunked` to write samples

    Args:
        backend: the Backend instance which created this SndWriter
        sr: the samplerate
        outfile: the file to wite
        encoding: the encoding of the file (*pcmXX*, *floatXX*, where *XX* represents
            the bits per sample)
        fileformat: the fileformat, only needed if the format indicated by the
            extension in outfile should be overridden
        metadata: a dict ``{str: str}``.


    **Metadata Possible Keys**:

    * title
    * comment
    * artist
    * album
    * tracknumber
    * software

    Example
    =======

    >>> from sndfileio import *
    >>> writer = sndwrite_chunked("out.flac", 44100)
    # writer is a SndWriter
    >>> for buf in sndread_chunked("in.flac"):
    ...     # do some processing, like changing the gain
    ...     buf *= 0.5
    ...     writer.write(buf)
    """

    def __init__(self, sr: int, outfile: str, encoding: str,
                 fileformat: str = None, bitrate=0,
                 backend=None,
                 metadata: Dict[str, str] = None) -> None:
        if metadata:
            for key in metadata:
                if key not in util.metadata_possible_keys:
                    raise KeyError(f"Metadata key {key} unknown. Possible keys: "
                                   f"{util.metadata_possible_keys}")
        self.sr: int = sr
        self.outfile: str = outfile
        self.encoding: str = encoding
        self.metadata: Opt[Dict[str, str]] = metadata
        self.bitrate = bitrate
        self.fileformat = fileformat or util.fileformat_from_ext(os.path.splitext(outfile)[1])
        self._backend = backend
        self._file = None

    def __call__(self, frames: np.ndarray) -> None:
        """
        Write the given sample data.

        The first array will determine the number of channels to write

        Args:
            frames (np.ndarray): the samples to write
        """
        return self.write(frames)

    def write(self, frames: np.ndarray) -> None:
        """
        Write the given sample data.

        The first array will determine the number of channels to write

        Args:
            frames (np.ndarray): the samples to write
        """
        pass

    def close(self) -> None:
        """
        Explicitely close this file
        """
        if self._file is not None:
            self._file.close()
        self._file = None

    def __enter__(self) -> SndWriter:
        return self

    def __exit__(self) -> None:
        self.close()

    @property
    def filetypes(self) -> List[str]:
        return self._backend.filetypes_write
