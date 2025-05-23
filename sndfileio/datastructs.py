from __future__ import annotations
import numpy as np
from . import util
import os

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    sample_t = tuple[np.ndarray, int]


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
                 encoding: str, fileformat: str, metadata: dict[str, str] | None = None,
                 extrainfo: dict[str, Any] | None = None,
                 bitrate: int = 0
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

    def __init__(self,
                 sr: int,
                 outfile: str,
                 encoding: str,
                 fileformat='',
                 bitrate=0,
                 backend=None,
                 metadata: dict[str, str] | None = None
                 ) -> None:
        if metadata:
            for key in metadata:
                if key not in util.metadata_possible_keys:
                    raise KeyError(f"Metadata key {key} unknown. Possible keys: "
                                   f"{util.metadata_possible_keys}")
        self.sr: int = sr
        self.outfile: str = outfile
        self.encoding: str = encoding
        self.metadata: dict[str, str] = metadata or {}
        self.bitrate = bitrate
        self.fileformat = fileformat or util.fileformat_from_ext(os.path.splitext(outfile)[1]) or 'wav'
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
        if self._file:
            self._file.close()
        self._file = None

    def __enter__(self) -> SndWriter:
        return self

    def __exit__(self) -> None:
        self.close()

    @property
    def filetypes(self) -> list[str]:
        if self._backend is None:
            return []
        return self._backend.filetypes_write
