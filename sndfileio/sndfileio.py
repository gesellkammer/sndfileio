from __future__ import annotations
import os as _os
import numpy as np
import importlib.util
import logging
import pysndfile
from . import backend_miniaudio

from .util import numchannels
from .datastructs import SndInfo, Sample
from typing import Dict, Tuple, Union, Iterator, Optional as Opt, List, Callable, Type
import numpyx

logger = logging.getLogger("sndfileio")

__all__ = [
    "sndread",
    "sndread_chunked",
    "sndget",
    "sndinfo",
    "sndwrite",
    "sndwrite_like",
    "sndwrite_chunked",
    "sndwrite_chunked_like",
    "bitdepth",
    "asmono",
    "getchannel",
    "SndInfo",
    "SndWriter"
]


_metadata_possible_keys = {'comment', 'title', 'artist', 'album', 'tracknumber', 'software'}

_encodings_for_format = {
    'wav': ['pcm16', 'pcm24', 'pcm32', 'float32', 'float64'],
    'aiff': ['pcm16', 'pcm24', 'pcm32', 'float32', 'float64'],
    'flac': ['pcm16', 'pcm24'],
    'mp3': ['pcm16', 'pcm24'],
    'ogg': ['pcm16', 'pcm24']
}

class FormatNotSupported(Exception):
    pass


def _is_package_installed(pkg):
    return importlib.util.find_spec(pkg) is not None


class SndfileError(IOError):
    pass


#######################################
#
#             Utilities
#
########################################


def _chunks(start:int, end:int, step:int) -> Iterator[Tuple[int, int]]:
    pos = start
    last_full = end - step
    while pos < last_full:
        yield pos, step
        pos += step
    yield pos, end - pos


########################################
#
#                API
#
########################################


_known_fileformats = _encodings_for_format.keys()


def _fileformat_from_ext(ext: str) -> str:
    """
    Deduces the file format from the extension

    The returned file format is always a lowercase string,
    one of "wav", "aiff", "flac", "ogg", "wv", "mp3"

    Examples
    --------

    >>> _fileformat_from_ext(("out.wav"))
    wav
    >>> _fileformat_from_ext(("foo.FLAC"))
    flac
    >>> _fileformat_from_ext(("bar.aif"))
    aiff

    """
    if ext.startswith("."):
        ext = ext[1:]
    ext = ext.lower()
    if ext == "aif":
        return "aiff"
    assert ext in _known_fileformats
    return ext


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

    >>> writer = sndwrite_chunked("out.flac", 44100)
    # writer is a SndWriter
    >>> for buf in sndread_chunked("in.flac"):
    ...     # do some processing, like changing the gain
    ...     buf *= 0.5
    ...     writer.write(buf)
    """
    def __init__(self, backend, sr:int, outfile:str, encoding:str,
                 fileformat: str = None,
                 metadata: Dict[str, str]=None) -> None:
        if metadata:
            for key in metadata:
                if key not in _metadata_possible_keys:
                    raise KeyError(f"Metadata key {key} unknown. Possible keys: "
                                   f"{_metadata_possible_keys}")
        self.sr:int = sr
        self.outfile:str = outfile
        self.encoding:str = encoding
        self.metadata: Opt[Dict[str, str]] = metadata
        self.fileformat = fileformat or _fileformat_from_ext(_os.path.splitext(outfile)[1])
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


def sndread(path:str, start:float=0, end:float=0) -> Sample:
    """
    Read a soundfile as a numpy array.

    This is a float array defined between -1 and 1, independently of the format
    of the soundfile

    Args:
        path: the path to read
        start: the time to start reading
        end: the time to end reading (0=until the end)

    Returns:
        a namedtuple (samples:ndarray[dtype=float], sr:int)

    Example
    ~~~~~~~

    .. code::

        # Normalize and save as flac
        from sndfileio import sndread, sndwrite
        samples, sr = sndread("in.wav")
        maxvalue = max(samples.max(), -samples.min())
        samples *= 1/maxvalue
        sndwrite(samples, sr, "out.flac")
    """
    backend = _get_backend(path)
    if not backend:
        raise RuntimeError(f"No backend available to read {path}")
    logger.debug(f"sndread: using backend {backend.name}")
    return backend.read(path, start=start, end=end)


def sndread_chunked(path:str, chunksize:int=2048, start:float=0., stop:float=0.
                    ) -> Iterator[np.ndarray]:
    """
    Read a soundfile in chunks

    Args:
        path: the path to read
        chunksize: the chunksize, in samples
        start: time to skip before reading
        stop: time to stop reading (0=end of file)

    Returns:
        a generator yielding numpy arrays (float64) of at most `chunksize` frames

    Example
    ~~~~~~~

    .. code::

        >>> with sndwrite_chunked("out.flac", 44100) as writer:
        ...     for buf in sndread_chunked("in.flac"):
        ...         # do some processing, like changing the gain
        ...         buf *= 0.5
        ...         writer.write(buf)

    """
    backend = _get_backend(path, key=lambda backend: backend.can_read_chunked)
    if backend:
        logger.debug(f"sndread_chunked: using backend {backend.name}")
        return backend.read_chunked(path, chunksize, start=start, stop=stop)
    else:
        raise SndfileError("chunked reading is not supported by the "
                           "available backends")


def sndinfo(path:str) -> SndInfo:
    """
    Get info about a soundfile. Returns a :class:`SndInfo`

    Args:
        path: the path to a soundfile

    Returns:
        a :class:`SndInfo` (attributes: **samplerate**: `int`, *nframes*: `int`, *channels*: `int`,
        **encoding**: `str`, **fileformat**: `str`, **metadata**: `dict`)

    Example
    ~~~~~~~
    ::

        >>> from sndfileio import *
        >>> info = sndinfo("sndfile.wav")
        >>> print(f"Duration: {info.duration}s, samplerate: {info.samplerate}")
        Duration: 0.4s, samplerate: 44100

    """
    backend = _get_backend(path)
    if not backend:
        raise FormatNotSupported("sndinfo: no backend supports this filetype")
    logger.debug(f"sndinfo: using backend {backend.name}")
    return backend.getinfo(path)


def sndget(path:str, start:float=0, end:float=0) -> Tuple[np.ndarray, SndInfo]:
    """
    Read a soundfile and its metadata

    Args:
        path: the path to read
        start: the time to start reading
        end: the time to end reading (0=until the end)

    Returns:
        a tuple (samples: np.ndarray, :class:`SndInfo`)

    Example
    ~~~~~~~

        # Normalize and save as flac, keeping the metadata
        >>> from sndfileio import *
        >>> samples, info = sndget("in.wav")
        >>> maxvalue = max(samples.max(), -samples.min())
        >>> samples *= 1/maxvalue
        >>> sndwrite("out.flac", samples, info.samplerate, metadata=info.metadata)
    """
    samples, sr = sndread(path, start=start, end=end)
    return samples, sndinfo(path)


def sndwrite(outfile:str, samples:np.ndarray, sr:int, encoding:str='auto',
             fileformat:str=None, normalize_if_clipping=True,
             metadata: Dict[str, str]=None) -> None:
    """
    Write all samples to a soundfile.

    Args:
        outfile: The name of the outfile. the extension will determine
            the file-format. The formats supported depend on the available
            backends.
        samples: Array-like. the actual samples. The shape determines the
            number of channels of outfile. For 1 channel, ``shape=(nframes,)`` or
            ``shape=(nframes, 1)``. For multichannel audio, ``shape=(nframes, nchannels)``.
        sr: Sampling-rate
        encoding: one of "auto", "pcm16", "pcm24", "pcm32", "pcm64", "float32", "float64"
        fileformat: if given, will override the format indicated by the extension
        normalize_if_clipping: prevent clipping by normalizing samples before
            writing. This only makes sense when writing pcm data
        metadata: a dict of str:str, with possible keys 'title', 'comment', 'artist',
            'album', 'tracknumber', 'software' (the creator of a soundfile)

    .. note::

        Not all file formats support all encodings. Raises :class:`SndfileError`
        if the format does not support the given encoding.
        If set to 'auto', an encoding will be selected based on the
        file-format and on the data. The bitdepth of the data is
        measured, and if the file-format supports it, it will be used.
        For bitdepths of 8, 16 and 24 bits, a PCM encoding will be used.
        For a bitdepth of 32 bits, a FLOAT encoding will be used,
        or the next lower supported encoding

    Example
    ~~~~~~~
    ::

        # Normalize and save as flac
        >>> samples, sr = sndread("sndfile.wav")
        >>> maxvalue = max(samples.max(), -samples.min())
        >>> samples *= 1/maxvalue
        >>> sndwrite("out.flac", samples, sr)

    """
    if not fileformat:
        fileformat = _fileformat_from_ext(_os.path.splitext(outfile)[1])
    if encoding in ('auto', None):
        encoding = guess_encoding(samples, fileformat)
    if encoding.startswith('pcm') and normalize_if_clipping:
        clipping = ((samples > 1).any() or (samples < -1).any())
        if clipping:
            maxvalue = max(samples.max(), abs(samples.min()))
            logger.warning(f"Clipping found when writing pcm data to {outfile}")
            samples = samples / maxvalue
    backend = _get_write_backend(fileformat)
    if not backend:
        raise SndfileError(f"No backend found to support the given format: {fileformat}")
    logger.debug(f"sndwrite: using backend {backend.name}")
    writer = backend.writer(sr=sr, outfile=outfile, encoding=encoding, metadata=metadata,
                            fileformat=fileformat)
    if not writer:
        raise SndfileError(f"Could not write to {outfile} with backend {backend.name}")
    return writer.write(samples)


def sndwrite_chunked(outfile:str, sr: int, encoding: str='auto', fileformat:str=None,
                     metadata: Dict[str, str]=None) -> SndWriter:
    """
    Opens a file for writing and returns a SndWriter

    The :meth:`~SndWriter.write` method of the returned :class:`SndWriter` can be
    called to write samples to the file

    Raises SndfileError if the format does not support the given encoding.

    Not all file formats support all encodings. If set to 'auto', an encoding will
    be selected based on the file-format and on the data. The bitdepth of the data is
    measured, and if the file-format supports it, it will be used.
    For bitdepths of 8, 16 and 24 bits, a PCM encoding will be used.
    For a bitdepth of 32 bits, a FLOAT encoding will be used,
    or the next lower supported encoding

    Args:
        outfile: The name of the outfile. the extension will determine the file-format.
            The formats supported depend on the available backends.
        sr: Sampling-rate
        encoding: one of 'auto', 'pcm16', 'pcm24', 'pcm32', 'float32', 'float64'.
        fileformat: needed only if the format cannot be determined from the extension
            (for example, if saving to an outfile with a non-traditional extension)
        metadata: a dict ``{str:str}`` with possible keys: 'comment', 'title', 'artist',
            'album', 'tracknumber', 'software' (the creator of a soundfile)

    Returns:
        a :class:`~sndfileio.SndWriter`, whose method :meth:`~sndfileio.SndWriter.write` can
        be called to write samples

    Example
    ~~~~~~~

        >>> with sndwrite_chunked("out.flac", 44100) as writer:
        ...     for buf in sndread_chunked("in.flac"):
        ...         # do some processing, like changing the gain
        ...         buf *= 0.5
        ...         writer.write(buf)

    """
    backends = [backend for backend in _get_backends() if backend.can_write_chunked]
    if not backends:
        raise SndfileError("No backend found to support the given format")
    backend = min(backends, key=lambda backend:backend.priority)
    logger.debug(f"sndwrite_chunked: using backend {backend.name}")
    return backend.writer(outfile, sr, encoding, metadata=metadata, fileformat=fileformat)


def asmono(samples:np.ndarray, channel:Union[int, str]=0) -> np.ndarray:
    """
    convert samples to mono if they are not mono already.

    The returned array will always have the shape (numframes,)

    Args:
        samples: the multichannel samples
        channel: the channel number to use, or 'mix' to mix-down all channels

    Returns:
        the samples as one mono channel
    """
    if numchannels(samples) == 1:
        return samples

    if isinstance(channel, int):
        return samples[:, channel]
    elif channel == 'mix':
        return _mix(samples, scale_by_numchannels=True)
    else:
        raise ValueError("channel has to be an integer indicating a channel,"
                         " or 'mix' to mix down all channels")


def getchannel(samples: np.ndarray, channel:int) -> np.ndarray:
    """
    Returns a view into a channel of samples.

    Args:
        samples: a numpy array representing the audio data
        channel: the channel to extract (channels begin with 0)

    Returns:
        the channel specified, as a numpy array

    Example
    ~~~~~~~
    ::

        # Read a stereo file, atenuate one channel
        >>> stereo, sr = sndread("stereo.wav")
        >>> ch0 = getchannel(stereo, 0)
        >>> ch0 *= 0.5
        >>> sndwrite_like(stereo, "stereo.wav", "out.wav")

    """
    N = numchannels(samples)
    if channel > (N - 1):
        raise ValueError(f"channel {channel} out of range (max. {N} channels)")
    if N == 1:
        return samples
    return samples[:, channel]


def bitdepth(data:np.ndarray, snap:bool=True) -> int:
    """
    returns the number of bits actually used to represent the data.

    Args:
        data: a numpy.array (mono or multi-channel)
        snap: snap to 8, 16, 24 or 32 bits.

    Returns:
        the bits needed to represent the data
    """
    data = asmono(data)
    maxitems = min(4096, data.shape[0])
    maxbits = max(x.as_integer_ratio()[1]
                  for x in data[:maxitems]).bit_length()
    if snap:
        if maxbits <= 8:
            maxbits = 8
        elif maxbits <= 16:
            maxbits = 16
        elif maxbits <= 24:
            maxbits = 24
        elif maxbits <= 32:
            maxbits = 32
        else:
            maxbits = 64
    return maxbits


def sndwrite_like(outfile:str, samples:np.ndarray, likefile:str, sr:int=None,
                  metadata: Dict[str, str]=None) -> None:
    """
    Write samples to outfile with samplerate/encoding taken from likefile

    Args:
        samples: the samples to write
        likefile: the file to use as a reference for sr, format and encoding
        outfile: the file to write to
        sr: sample rate can be overridden
        metadata: a dict {str:str}, overrides metadata in `likefile`.


    Example
    ~~~~~~~

    .. code-block:: python

        # Read a file, apply a fade-in of 0.5 seconds, save it
        import numpy as np
        samples, sr = sndread("stereo.wav")
        fadesize = int(0.5*sr)
        ramp = np.linspace(0, 1, fadesize))
        samples[:fadesize, 0] *= ramp
        samples[:fadesize, 1] *= ramp
        sndwrite_like(samples, "stereo.wav", "out.wav")

    """
    info = sndinfo(likefile)
    sndwrite(outfile=outfile, samples=samples, sr=sr or info.samplerate,
             encoding=info.encoding, metadata=metadata or info.metadata)


def sndwrite_chunked_like(likefile:str, outfile:str, sr:int=None,
                          metadata: Dict[str, str] = None
                          ) -> SndWriter:
    """
    Create a SndWriter with samplerate/format/encoding of the
    source file

    Args:
        likefile: the file to use as reference
        outfile: the file to open for writing
        sr: samplerate can be overridden
        metadata: a dict {str:str}, overrides metadata in `likefile`.

    Returns:
        a :class:`SndWriter`. Call :meth:`~SndWriter.write` on it to write to
        the file
    """
    info = sndinfo(likefile)
    return sndwrite_chunked(outfile=outfile, sr=sr or info.samplerate,
                            encoding=info.encoding, metadata=metadata or info.metadata)


############################################
#
#                BACKENDS
#
############################################

def _asbytes(s: Union[str, bytes]) -> bytes:
    if isinstance(s, bytes):
        return s
    return s.encode('ascii')


class _PySndfileWriter(SndWriter):
    _keyTable = {
        'comment': 'SF_STR_COMMENT',
        'title': 'SF_STR_TITLE',
        'artist': 'SF_STR_ARTIST',
        'album': 'SF_STR_ALBUM',
        'tracknumber': 'SF_STR_TRACKNUMBER',
        'software': 'SF_STR_SOFTWARE'
    }

    def _open_file(self, channels:int) -> None:
        if self.fileformat not in self.filetypes:
            raise ValueError(f"Format {self.fileformat} not supported by this backend")
        sndformat = self._backend._get_sndfile_format(self.fileformat, self.encoding)
        self._file = pysndfile.PySndfile(self.outfile, "w", sndformat, channels, self.sr)
        if self.metadata:
            for k, v in self.metadata.items():
                key = self._keyTable[k]
                self._file.set_string(key, _asbytes(v))

    def write(self, frames:np.ndarray) -> None:
        if self._file:
            self._file.write_frames(frames)
        else:
            nchannels = numchannels(frames)
            if self.encoding == 'auto':
                self.encoding = guess_encoding(frames, self.fileformat)
            self._open_file(nchannels)
            self.write(frames)

    def close(self):
        if self._file is None:
            raise IOError("Can't close, since this file was never open")
        self._file.writeSync()
        del self._file
        self._file = None


class Backend:
    def __init__(self, priority:int,
                 filetypes:List[str],
                 filetypes_write:List[str],
                 can_read_chunked:bool,
                 can_write_chunked:bool,
                 name:str,
                 supports_metadata:bool):
        self.priority = priority
        self.filetypes = filetypes
        self.filetypes_write = filetypes_write
        self.can_read_chunked = can_read_chunked
        self.can_write_chunked = can_write_chunked
        self.supports_metadata = supports_metadata
        self.name = name
        self._backend = None
        self._writer: Opt[Type[SndWriter]] = None

    def read(self, path:str, start:float=0, end:float=0) -> Sample:
        return NotImplemented

    def read_chunked(self, path:str, chunksize:int=2048, start:float=0., stop:float=0.
                     ) -> Iterator[np.ndarray]:
        return NotImplemented

    def getinfo(self, path:str) -> SndInfo:
        """ Get info about the soundfile given. Returns a SndInfo structure """
        return NotImplemented

    def is_available(self) -> bool:
        """ Is this backend available? """
        return _is_package_installed(self.name)

    def writer(self, outfile:str, sr:int, encoding:str, fileformat:str,
               metadata: Dict[str, str]=None) -> SndWriter:
        """ Open outfile for write with the given properties 

        Args:
            sr: samplerate
            outfile: the file to write
            encoding: the encoding used (pcm16, float32, etc)
            fileformat: the file format
            metadata: if given, a dict str:str. Allowed keys are: *title*, *comment*,
                *artist*, *tracknumber*, *album*, *software*

        Returns:
            a :class:`SndWriter` 
        """
        if self._writer is None:
            raise SndfileError("This backend does not support writing")
        return self._writer(self, sr=sr, outfile=outfile, encoding=encoding,
                            fileformat=fileformat, metadata=metadata)

    def check_write(self, fileformat:str, encoding:str) -> None:
        """ Check if we can write to outfile with the given encoding """
        if encoding not in _encodings_for_format[fileformat]:
            raise ValueError("Encoding not supported")
        if fileformat not in self.filetypes_write:
            raise ValueError(f"The given format {fileformat} is not supported by the "
                             f"{self.name} backend")

    def dump(self) -> None:
        """ Dump information about this backend """
        print(f"Backend: {self.name} (available: {self.is_available}, priority: {self.priority})")
        if self.filetypes:
            readtypes = ", ".join(self.filetypes)
            print(f"    read types : {readtypes}")
        if self.filetypes_write:
            writetypes = ", ".join(self.filetypes_write)
            print(f"    write types: {writetypes}")
        ok, notok = "OK", "--"
        readchunked  = ok if self.can_read_chunked else notok
        writechunked = ok if self.can_write_chunked else notok
        print(f"    sndread_chunked: {readchunked}    sndwrite_chunked: {writechunked}")


class _PySndfile(Backend):
    """
    A backend based in pysndfile

    """
    _keyTable = {'SF_STR_COMMENT': 'comment',
                 'SF_STR_TITLE': 'title',
                 'SF_STR_ARTIST': 'artist',
                 'SF_STR_ALBUM': 'album',
                 'SF_STR_TRACKNUMBER': 'tracknumber',
                 'SF_STR_SOFTWARE': 'software'}

    def __init__(self, priority:int):
        super().__init__(
                priority  = priority,
                filetypes = ["aif", "aiff",  "wav", "flac", "ogg", "wav64", "caf", "raw"],
                filetypes_write = ["aif", "aiff",  "wav", "flac", "ogg", "wav64", "caf", "raw"],
                can_read_chunked = True,
                can_write_chunked = True,
                name = 'pysndfile',
                supports_metadata= True
        )
        self._writer = _PySndfileWriter

    def read(self, path:str, start:float=0, end:float=0) -> Sample:
        snd = pysndfile.PySndfile(path)
        sr = snd.samplerate()
        samp_start = int(start * sr)
        samp_end = int(end * sr) if end > 0 else snd.frames()
        if samp_start:
            snd.seek(samp_start)
        data = snd.read_frames(samp_end - samp_start)
        return Sample(data, sr)

    def read_chunked(self, path:str, chunksize:int=2048, start:float=0., stop:float=0.
                     ) -> Iterator[np.ndarray]:
        snd = pysndfile.PySndfile(path)
        sr = snd.samplerate()
        if start:
            snd.seek(int(start*snd.samplerate()))
        firstframe = int(sr * start)
        if stop == 0:
            lastframe = snd.frames()
        else:
            lastframe = int(sr * stop)

        for pos, nframes in _chunks(0, lastframe - firstframe, chunksize):
            yield snd.read_frames(nframes)

    def getinfo(self, path:str) -> SndInfo:
        snd = pysndfile.PySndfile(path)
        metadataraw: Dict[str, bytes] = snd.get_strings()
        if metadataraw:
            metadata = {}
            extrainfo = {}
            for k, v in metadataraw.items():
                if k not in self._keyTable:
                    extrainfo[k] = v
                else:
                    metadata[self._keyTable[k]] = v
        else:
            metadata = None
            extrainfo = None

        return SndInfo(snd.samplerate(), snd.frames(), snd.channels(),
                       snd.encoding_str(), snd.major_format_str(),
                       metadata=metadata, extrainfo=extrainfo)

    def write(self, data:np.ndarray, sr:int, outfile:str, encoding:str) -> None:
        self.check_write(outfile, encoding)
        ext = _os.path.splitext(outfile)[1].lower()
        fmt = self._get_sndfile_format(ext, encoding)
        snd = pysndfile.PySndfile(outfile, mode='w', format=fmt,
                                  channels=numchannels(data), samplerate=sr)
        snd.write_frames(data)
        snd.writeSync()

    def _get_sndfile_format(self, fileformat: str, encoding: str) -> int:
        """
        Construct a pysndfile format id from fileformat and encoding

        Args:
            fileformat: the fileformat, one of 'wav', 'aiff', etc
            encoding: one of *pcmXX* or *floatXX* (where *XX *is the number of bits/sample,
                one of 16, 24, 32, 64)

        Returns:
            the pysndfile format id
        """
        assert fileformat in self.filetypes
        fmt, bits = encoding[:-2], int(encoding[-2:])
        assert fmt in ('pcm', 'float') and bits in (8, 16, 24, 32, 64)
        if fileformat == 'aif':
            fileformat = 'aiff'
        fmt = f"{fmt}{bits}"
        return pysndfile.construct_format(fileformat, fmt)

    def detect_format(self, path:str) -> Opt[str]:
        f = pysndfile.PySndfile(path)
        return pysndfile.fileformat_id_to_name.get(f.format())


class _Miniaudio(Backend):

    def __init__(self, priority):
        super().__init__(
                priority=priority,
                filetypes= ['mp3'],
                filetypes_write = ['mp3'],
                can_read_chunked = True,
                can_write_chunked = False,
                name = 'miniaudio',
                supports_metadata=False
        )

    def getinfo(self, path:str) -> SndInfo:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            return backend_miniaudio.mp3info(path)
        else:
            raise FormatNotSupported(f"format {ext} is not supported")

    def read(self, path: str, start:float=0., end:float=0.) -> Sample:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            return backend_miniaudio.mp3read(path, start=start, end=end)
        else:
            raise FormatNotSupported(f"This backend does not support {ext} format")

    def read_chunked(self, path:str, chunksize:int=2048, start:float=0., stop:float=0.
                     ) -> Iterator[np.ndarray]:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            return backend_miniaudio.mp3read_chunked(path, chunksize=chunksize,
                                                     start=start, stop=stop)
        else:
            raise FormatNotSupported(f"This backend does not support {ext} format")



BACKENDS: List[Backend] = [
    _PySndfile(priority=0),
    _Miniaudio(priority=10),
]


def report_backends():
    for b in BACKENDS:
        if b.is_available():
            b.dump()
        else:
            print(f"Backend {b.name} NOT available")


#   HELPERS ------------------------------------


def as_float_array(data:np.ndarray, encoding:str) -> np.ndarray:
    """
    Convert (if necessary) an array containing pcm (integer) samples
    to float64 between -1:1

    Args:
        data: the samples to convert
        encoding: the encoding of data (one of 'float32', 'pcm24', 'pcm16')

    Returns:
        data represented as a float numpy array
    """
    assert (data > 0).any()
    if encoding == 'float32':
        return data
    elif encoding == 'pcm24':
        return data / (2.0 ** 31)
    elif encoding == 'pcm16':
        return data / (2.0 ** 15)
    else:
        raise ValueError("encoding not understood")


def numpy24to32bit(data:np.ndarray, bigendian:bool=False) -> np.ndarray:
    """
    Convert a 24-bit pcm to 32-bit pcm samples

    Args:
        data: a ubyte array of shape = (size,) (interleaved channels if multichannel)
        bigendian: is the data represented as big endian?
    """
    target_size = int(data.shape[0] * 4 / 3.)
    target = np.zeros((target_size,), dtype=np.ubyte)
    if not bigendian:
        target[3::4] = data[2::3]
        target[2::4] = data[1::3]
        target[1::4] = data[0::3]
    else:
        target[1::4] = data[2::3]
        target[2::4] = data[1::3]
        target[3::4] = data[0::3]
    targetraw = target.tobytes()
    del target
    return np.fromstring(targetraw, dtype=np.int32)


_cache = {}


def _get_backends() -> List[Backend]:
    backends = _cache.get('backends')
    if not backends:
        _cache['backends'] = backends = [b for b in BACKENDS if b.is_available()]
    return backends


def _detect_format(path:str) -> Opt[str]:
    ext = _os.path.splitext(path)[1][1:].lower() if path else None
    if ext in _known_fileformats:
        return ext
    import filetype
    ext = filetype.guess_extension(path)
    if ext in _known_fileformats:
        return ext
    return None


def _get_backend(path:str=None, key:Callable[[Backend], bool]=None) -> Opt[Backend]:
    """
    Get available backends to read/write the file given

    Args:
        path: the file to read/write
        key: a function (backend) -> bool signaling if the backend
             is suitable for a specific task

    Example
    ~~~~~~~
    ::

        # Get available backends which can read in chunks
        >>> backend = _get_backend('file.flac',
        ...                        key=lambda backend:backend.can_read_chunked())
    """
    filetype = _detect_format(path)
    backends = _get_backends()
    if key:
        backends = [b for b in backends if key(b)]
    if filetype:
        backends = [b for b in backends if filetype in b.filetypes]
    if backends:
        return min(backends, key=lambda backend: backend.priority)
    return None


def _get_write_backend(fileformat:str) -> Opt[Backend]:
    assert fileformat in _known_fileformats
    backends = _get_backends()
    if not backends:
        raise SndfileError("No available backends for writing")
    backends = [b for b in backends if fileformat in b.filetypes_write]
    if backends:
        return min(backends, key=lambda backend: backend.priority)
    return None


def _mix(samples:np.ndarray, scale_by_numchannels:bool=True) -> np.ndarray:
    summed = samples.sum(0)
    if scale_by_numchannels:
        summed *= (1 / numchannels(samples))
    return summed


def _samples_out_of_range(data:np.ndarray) -> bool:
    for i in range(numchannels(data)):
        ch = getchannel(data, i)
        if numpyx.any_less_than(ch, -1) or numpyx.any_greater_than(ch, 1):
            return True
    return False


def guess_encoding(data:np.ndarray, fmt: str) -> str:
    """
    Guess the encoding for data based on the format. For each format
    an encoding is selected which is able to represent the data without
    loss. In the case of wav/aiff, if there is sample data outside of the
    -1:1 range, a float32 encoding is chosen, since any pcm representation
    would result in out of range samples.

    Args:
        data: the samples
        fmt: on of 'wav', 'aiff', 'flac'

    Returns:
        the encoding, one of 'pcm16', 'pcm24', 'float32'
    """
    if fmt in ('wav', 'aif', 'aiff'):
        if _samples_out_of_range(data):
            return 'float32'
        maxbits = min(32, bitdepth(data, snap=True))
        encoding = {
            16: 'pcm16',
            24: 'pcm24',
            32: 'flpat32',
        }.get(maxbits, 'float32')
    elif fmt == "flac":
        maxbits = min(24, bitdepth(data, snap=True))
        encoding = {
            16: 'pcm16',
            24: 'pcm24',
        }.get(maxbits, 'pcm24')
    else:
        raise FormatNotSupported(f"The format {fmt} is not supported")
    assert encoding in ('pcm16', 'pcm24', 'float32')
    return encoding
