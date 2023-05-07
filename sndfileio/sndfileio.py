from __future__ import annotations
import os as _os
import sys as _sys
import numpy as np
from importlib.util import find_spec as _find_spec
import logging

try:
    import soundfile as _soundfile
except (IOError, ImportError) as e:
    if 'sphinx' in _sys.modules:
        from sphinx.ext.autodoc.mock import _MockObject
        _soundfile = _MockObject()
    else:
        raise e

from . import util
from .datastructs import SndInfo, SndWriter

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pysndfile
    from .datastructs import sample_t
    from typing import Dict, Tuple, Union, Iterator, Optional as Opt, List, Callable, Type


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
    "mp3write",
    "SndInfo",
    "SndWriter"
]


class FormatNotSupported(Exception):
    pass


def _is_package_installed(pkg):
    return _find_spec(pkg) is not None


class SndfileError(IOError):
    pass


def _normalize_path(path: str) -> str:
    return _os.path.expanduser(path)


def sndread(path: str, start: float = 0, end: float = 0) -> sample_t:
    """
    Read a soundfile as a numpy array.

    This is a float array defined between -1 and 1, independently of the format
    of the soundfile

    Args:
        path: (str) the path to read
        start: (float) the time to start reading
        end: (float) the time to end reading (0=until the end)

    Returns:
        a tuple (samples:ndarray[dtype=float], sr: int)

    Example
    ~~~~~~~

    ::

        # Normalize and save as flac
        from sndfileio import sndread, sndwrite
        samples, sr = sndread("in.wav")
        maxvalue = max(samples.max(), -samples.min())
        samples *= 1/maxvalue
        sndwrite(samples, sr, "out.flac")
    """
    path = _normalize_path(path)
    if not _os.path.exists(path):
        raise IOError(f"File not found: {path}")
    backend = _get_backend(path)
    if not backend:
        raise RuntimeError(f"No backend available to read {path}")
    logger.debug(f"sndread: using backend {backend.name}")
    return backend.read(path, start=start, end=end)


def sndread_chunked(path: str, chunksize: int = 2048, start: float = 0., stop: float = 0.
                    ) -> Iterator[np.ndarray]:
    """
    Read a soundfile in chunks

    Args:
        path: (str) the path to read
        chunksize: (int) the chunksize, in samples
        start: (float) time to skip before reading
        stop: (float) time to stop reading (0=end of file)

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
    path = _normalize_path(path)
    backend = _get_backend(path, key=lambda backend: backend.can_read_chunked)
    if backend:
        logger.debug(f"sndread_chunked: using backend {backend.name}")
        return backend.read_chunked(path, chunksize, start=start, stop=stop)
    else:
        raise SndfileError("chunked reading is not supported by the "
                           "available backends")


def sndinfo(path: str) -> SndInfo:
    """
    Get info about a soundfile. Returns a :class:`SndInfo`

    Args:
        path: (str) the path to a soundfile

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
    path = _normalize_path(path)
    backend = _get_backend(path)
    if not backend:
        raise FormatNotSupported("sndinfo: no backend supports this filetype")
    logger.debug(f"sndinfo: using backend {backend.name}")
    return backend.getinfo(path)


def sndget(path: str, start: float = 0, end: float = 0) -> Tuple[np.ndarray, SndInfo]:
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
    ::

        # Normalize and save as flac, keeping the metadata
        from sndfileio import *
        samples, info = sndget("in.wav")
        maxvalue = max(samples.max(), -samples.min())
        samples *= 1/maxvalue
        sndwrite("out.flac", samples, info.samplerate, metadata=info.metadata)

    """
    path = _normalize_path(path)
    backend = _get_backend(path)
    if not backend:
        raise RuntimeError(f"No backend available to read {path}")
    logger.debug(f"sndread: using backend {backend.name}")
    return backend.read_with_info(path, start=start, end=end)


def _resolve_encoding(outfile: str, fileformat: Opt[str], encoding: Opt[str],
                      samples: Opt[np.ndarray] = None
                      ) -> Tuple[str, str]:
    if not fileformat:
        fileformat = util.fileformat_from_ext(_os.path.splitext(outfile)[1])
    if encoding == 'auto':
        if samples:
            encoding = util.guess_encoding(samples, fileformat)
        else:
            encoding = util.default_encoding(fileformat)
    elif encoding == 'default' or encoding is None:
        encoding = util.default_encoding(fileformat)
    return fileformat, encoding


def sndwrite(outfile: str, samples: np.ndarray, sr: int, encoding='default',
             fileformat: str = None, normalize_if_clipping=True,
             metadata: Dict[str, str] = None, **options
             ) -> None:
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
        encoding: one of "pcm16", "pcm24", "pcm32", "pcm64", "float32", "float64",
            "auto" to guess an encoding and "default" to use the default encoding
            for the given format.
        fileformat: if given, will override the format indicated by the extension
        normalize_if_clipping: prevent clipping by normalizing samples before
            writing. This only makes sense when writing pcm data
        metadata: a dict of str: str, with possible keys 'title', 'comment', 'artist',
            'album', 'tracknumber', 'software' (the creator of a soundfile)
        options: available options depend on the fileformat.

    .. admonition:: Options

        **mp3**:
            - `bitrate`: bitrate in Kb/s (int, default=160)
            - `quality`: 1-7, 1 is highest and 7 is fastest, default=3

        For lossless formats, like `wav`, `aif`, `flac`, etc., **there are no extra options**

    .. note::

        Not all file formats support all encodings. Raises :class:`SndfileError`
        if the format does not support the given encoding.
        If set to 'auto', an encoding will be selected based on the
        file-format and on the data. The bitdepth of the data is
        measured, and if the file-format supports it, it will be used.
        For bitdepths of 8, 16 and 24 bits, a PCM encoding will be used.
        For a bitdepth of 32 bits, a FLOAT encoding will be used,
        or the next lower supported encoding.
        If 'default' is given as encoding, the default encoding for the format
        is used

    =======    =================
    Format     Default encoding
    =======    =================
     wav        float32
     aif        float32
     flac       pcm24
     mp3        pcm16
    =======    =================


    Example
    ~~~~~~~
    ::

        # Normalize and save as flac
        >>> samples, sr = sndread("sndfile.wav")
        >>> maxvalue = max(samples.max(), -samples.min())
        >>> samples *= 1/maxvalue
        >>> sndwrite("out.flac", samples, sr)

    """
    outfile = _normalize_path(outfile)
    fileformat, encoding = _resolve_encoding(outfile, samples=samples, fileformat=fileformat,
                                             encoding=encoding)
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
                            fileformat=fileformat, **options)
    if not writer:
        raise SndfileError(f"Could not write to {outfile} with backend {backend.name}")
    return writer.write(samples)


def sndwrite_chunked(outfile: str, sr: int, encoding='auto', fileformat: str = None,
                     metadata: Dict[str, str] = None, **options
                     ) -> SndWriter:
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
        metadata: a dict ``{str: str}`` with possible keys: 'comment', 'title', 'artist',
            'album', 'tracknumber', 'software' (the creator of a soundfile)
        options: available options depend on the fileformat.
            For mp3, options are: bitrate (int, default=128) and quality (1-7, where
            1 is highest and 7 is fastest, default=2)
            For non-destructive formats, like wav, aif, flac, etc., there are no extra options

    Returns:
        a :class:`~sndfileio.SndWriter`, whose method :meth:`~sndfileio.SndWriter.write` can
        be called to write samples

    Example
    ~~~~~~~

    ::

        from snfileio import *
        with sndwrite_chunked("out.flac", 44100) as writer:
            for buf in sndread_chunked("in.flac"):
                # do some processing, like changing the gain
                buf *= 0.5
                writer.write(buf)

    """
    outfile = _normalize_path(outfile)
    fileformat, encoding = _resolve_encoding(outfile, fileformat=fileformat,
                                             encoding=encoding)
    backends = [backend for backend in _get_backends()
                if backend.can_write_chunked and fileformat in backend.filetypes_write]
    if not backends:
        raise SndfileError(f"No backend found to support the format {fileformat}")
    backend = min(backends, key=lambda backend: backend.priority)
    logger.debug(f"sndwrite_chunked: using backend {backend.name}")
    return backend.writer(outfile, sr, encoding, metadata=metadata, fileformat=fileformat,
                          **options)


def sndwrite_like(outfile: str, samples: np.ndarray, likefile: str, sr: int = None,
                  metadata: Dict[str, str] = None
                  ) -> None:
    """
    Write samples to outfile with samplerate/fileformat/encoding taken from likefile

    Args:
        outfile: the file to write to
        samples: the samples to write
        likefile: the file to use as a reference for sr, format and encoding
        sr: sample rate can be overridden
        metadata: a dict {str: str}, overrides metadata in `likefile`. Metadata
            is not merged, so if metadata is given, it substitutes the metadata
            in `likefile` completely. In order to merge it, do that beforehand
            If None is passed, the metadata in `likefile` is written to `outfile`

    .. note::

        The fileformat is always determined by `likefile`, even if the extension
        of `outfile` would result in a different format. For example, if `likefile`
        has a flac format but outfile has a .wav extension, the resulting file will
        be written in flac format.

    Example
    ~~~~~~~

    ::

        # Read a file, apply a fade-in of 0.5 seconds, save it
        import numpy as np
        from sndfileio import *
        samples, sr = sndread("stereo.wav")
        fadesize = int(0.5*sr)
        ramp = np.linspace(0, 1, fadesize))
        samples[:fadesize, 0] *= ramp
        samples[:fadesize, 1] *= ramp
        sndwrite_like(samples, "stereo.wav", "out.wav")

    """
    outfile = _normalize_path(outfile)
    info = sndinfo(likefile)
    ext = _os.path.splitext(outfile)[1]
    outfileformat = util.fileformat_from_ext(ext)
    if outfileformat != info.fileformat:
        logger.warning(f"Trying to save to a file with extension {ext}, but fileformat"
                       f"will be {info.fileformat}")
    sndwrite(outfile=outfile, samples=samples, sr=sr or info.samplerate,
             fileformat=info.fileformat, encoding=info.encoding,
             metadata=metadata or info.metadata)


def sndwrite_chunked_like(outfile: str, likefile: str, sr: int = None,
                          metadata: Dict[str, str] = None
                          ) -> SndWriter:
    """
    Create a SndWriter with samplerate/format/encoding of the
    source file

    Args:
        outfile: the file to open for writing
        likefile: the file to use as reference
        sr: samplerate can be overridden
        metadata: a dict {str: str}, overrides metadata in `likefile`. Metadata
            is not merged, so if metadata is given, it substitutes the metadata
            in `likefile` completely. In order to merge it, do that beforehand
            If None is passed, the metadata in `likefile` is written to `outfile`

    Returns:
        a :class:`SndWriter`. Call :meth:`~SndWriter.write` on it to write to
        the file
    """
    info = sndinfo(likefile)
    return sndwrite_chunked(outfile=outfile, sr=sr or info.samplerate,
                            encoding=info.encoding, metadata=metadata or info.metadata)


def mp3write(outfile: str, samples: np.ndarray, sr: int, bitrate=224, quality=3
             ) -> None:
    """
    Write all samples to outfile as mp3

    This is the same as::

        sndwrite(outfile, samples, sr, bitrate=224, quality=3)

    But in this case the arguments are explictely listed instead of being part of
    ``**options``

    Args:
        outfile: the outfile to write to
        samples: the samples to write (float samples in the range -1:1). They will
            be converted to int16 so any values outside the given range will clip
        sr: the samplerate
        bitrate: the bitrate to use, in Kb/s
        quality: the quality, a value between 1-7 (where 1 is highest and 7 is fastest)

    .. note::
        To write samples in chunks use sndwrite_chunked. `bitrate` and `quality` can
        be passed as **options.

    """
    outfile = _normalize_path(outfile)
    mp3backend = _BACKENDS['lameenc']
    if not mp3backend.is_available():
        raise RuntimeError("lameenc backend is not available")
    writer = mp3backend.writer(outfile=outfile, sr=sr, encoding='pcm16', fileformat='mp3',
                               bitrate=bitrate, quality=quality)
    writer.write(samples)


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

    def _open_file(self, channels: int) -> None:
        if self.fileformat not in self.filetypes:
            raise ValueError(f"Format {self.fileformat} not supported by this backend")
        sndformat = self._backend.get_sndfile_format(self.fileformat, self.encoding)
        self._file = self._backend.PySndfile(self.outfile, "w", sndformat, channels, self.sr)
        if self.metadata:
            for k, v in self.metadata.items():
                key = self._keyTable[k]
                self._file.set_string(key, _asbytes(v))

    def write(self, frames: np.ndarray) -> None:
        if not self._file:
            nchannels = util.numchannels(frames)
            if self.encoding == 'auto':
                self.encoding = util.guess_encoding(frames, self.fileformat)
            elif self.encoding == 'default':
                self.encoding = util.default_encoding(self.fileformat)
            self._open_file(nchannels)
        self._file.write_frames(frames)

    def close(self):
        if self._file is None:
            raise IOError("Can't close, since this file was never open")
        self._file.writeSync()
        del self._file
        self._file = None


class Backend:
    def __init__(self, priority: int,
                 filetypes: List[str],
                 filetypes_write: List[str],
                 can_read_chunked: bool,
                 can_write_chunked: bool,
                 name: str,
                 supports_metadata: bool):
        self.priority = priority
        self.filetypes = filetypes
        self.filetypes_write = filetypes_write
        self.can_read_chunked = can_read_chunked
        self.can_write_chunked = can_write_chunked
        self.supports_metadata = supports_metadata
        self.name = name
        self._backend = None
        self._writer: Opt[Type[SndWriter]] = None

    def read_with_info(self, path: str, start=0., end=0.
                       ) -> Tuple[np.ndarray, SndInfo]:
        return NotImplemented

    def read(self, path: str, start=0., end=0.) -> sample_t:
        samples, info = self.read_with_info(path=path, start=start, end=end)
        return samples, info.samplerate

    def read_chunked(self, path: str, chunksize=2048, start=0., stop=0.
                     ) -> Iterator[np.ndarray]:
        return NotImplemented

    def getinfo(self, path: str) -> SndInfo:
        """ Get info about the soundfile given. Returns a SndInfo structure """
        return NotImplemented

    def is_available(self) -> bool:
        """ Is this backend available? """
        return _is_package_installed(self.name)

    def writer(self, outfile: str, sr: int, encoding: str, fileformat: str,
               bitrate=0, metadata: Dict[str, str] = None, **options
               ) -> SndWriter:
        """ Open outfile for write with the given properties

        Args:
            sr: samplerate
            outfile: the file to write
            encoding: the encoding used (pcm16, float32, etc)
            fileformat: the file format
            bitrate: used when writing to a compressed file.
            metadata: if given, a dict str: str. Allowed keys are: *title*, *comment*,
                *artist*, *tracknumber*, *album*, *software*

        Returns:
            a :class:`SndWriter`
        """
        pass
        if self._writer is None:
            raise SndfileError(f"The backend {self.name} does not support writing")
        return self._writer(backend=self, sr=sr, outfile=outfile, encoding=encoding,
                            fileformat=fileformat, bitrate=bitrate, metadata=metadata,
                            **options)

    def check_write(self, fileformat: str, encoding: str) -> None:
        """ Check if we can write to outfile with the given encoding """
        if encoding not in util.encodings_for_format[fileformat]:
            raise ValueError("Encoding not supported")
        if fileformat not in self.filetypes_write:
            raise ValueError(f"The given format {fileformat} is not supported by the "
                             f"{self.name} backend")

    def dump(self) -> None:
        """ Dump information about this backend """
        print(f"Backend: {self.name} (available: {self.is_available()}, priority: {self.priority})")
        if self.filetypes:
            readtypes = ", ".join(self.filetypes)
            print(f"    read types : {readtypes}")
        if self.filetypes_write:
            writetypes = ", ".join(self.filetypes_write)
            print(f"    write types: {writetypes}")
        ok, notok = "OK", "--"
        readchunked = ok if self.can_read_chunked else notok
        writechunked = ok if self.can_write_chunked else notok
        print(f"    sndread_chunked: {readchunked}    sndwrite_chunked: {writechunked}")


class _Lameenc(Backend):

    def __init__(self, priority: int):
        super().__init__(priority=priority,
                         filetypes=[],
                         filetypes_write=['mp3'],
                         can_read_chunked=False,
                         can_write_chunked=True,
                         name='lameenc',
                         supports_metadata=False)

    def writer(self, outfile: str, sr: int, encoding: str, fileformat: str,
               metadata: Dict[str, str] = None, **options
               ) -> SndWriter:
        from . import backend_lameenc
        bitrate = options.pop('bitrate', 160)
        quality = options.pop('quality', 3)
        return backend_lameenc.LameencWriter(outfile=outfile, sr=sr, bitrate=bitrate,
                                             quality=quality)


class _SoundfileWriter(SndWriter):
    def _open_file(self, channels: int) -> None:
        if self.fileformat not in self.filetypes:
            raise ValueError(f"Format {self.fileformat} not supported by this backend")
        fmt, subtype = _Soundfile.get_format_and_subtype(self.fileformat, self.encoding)
        self._file = _soundfile.SoundFile(self.outfile, "w", format=fmt, subtype=subtype,
                                          channels=channels, samplerate=self.sr)
        if self.metadata:
            for key, value in self.metadata.items():
                setattr(self._file, key, value)

    def write(self, frames: np.ndarray) -> None:
        if not self._file:
            nchannels = util.numchannels(frames)
            if self.encoding == 'auto':
                self.encoding = util.guess_encoding(frames, self.fileformat)
            elif self.encoding == 'default':
                self.encoding = util.default_encoding(self.fileformat)
            self._open_file(nchannels)
        self._file.write(frames)

    def close(self):
        if self._file is None:
            return
        self._file.close()
        self._file = None


class _Soundfile(Backend):
    """
    A backend based on soundfile (https://pysoundfile.readthedocs.io)

    """
    # TODO: support metadata (either PR or via
    #  https://github.com/thebigmunch/audio-metadata)
    subtype_to_encoding = {
        'PCM_24': 'pcm24',
        'PCM_16': 'pcm16',
        'PCM_32': 'pcm32',
        'PCM_64': 'pcm64',
        'FLOAT': 'float32',
        'DOUBLE': 'float64',
        'VORBIS': 'vorbis'
    }

    encoding_to_subtype = {
        'pcm16': 'PCM_16',
        'pcm24': 'PCM_24',
        'float32': 'FLOAT',
        'float64': 'DOUBLE',
    }

    format_to_fileformat = {
        'WAV': 'wav',
        'WAVEX': 'wav',
        'AIFF': 'aiff',
        'FLAC': 'flac',
        'OGG': 'ogg'
    }

    # 'comment', 'title', 'artist',
    #             'album', 'tracknumber', 'software'
    metadata_keys = {'comment', 'title', 'artist', 'album', 'tracknumber', 'software'}

    def __init__(self, priority: int):
        super().__init__(
            priority=priority,
            filetypes=["aif", "aiff", "wav", "flac", "ogg"],
            filetypes_write=["aif", "aiff", "wav", "flac"],
            can_read_chunked=True,
            can_write_chunked=True,
            name='soundfile',
            supports_metadata=True
        )
        self._writer = _SoundfileWriter

    def read_with_info(self, path: str, start: float = 0, end: float = 0) -> Tuple[np.ndarray, SndInfo]:
        snd = _soundfile.SoundFile(path, 'r')
        info = self._getinfo(snd)
        samples = self._read(snd, start=start, end=end)
        return samples, info

    def _read(self, snd: _soundfile.SoundFile, start: float = 0, end: float = 0) -> np.ndarray:
        sr: int = snd.samplerate
        samp0 = int(start*sr)
        samp1 = int(end*sr) if end > 0 else snd.frames
        if samp0:
            snd.seek(samp0)
        return snd.read(samp1 - samp0)

    def read_chunked(self, path: str, chunksize=2048, start=0., stop=0.
                     ) -> Iterator[np.ndarray]:
        snd = _soundfile.SoundFile(path, 'r')
        sr = snd.samplerate
        if start:
            snd.seek(int(start*snd.samplerate))
        firstframe = int(sr * start)
        lastframe = snd.frames if stop == 0 else int(sr*stop)
        for pos, nframes in util.chunks(0, lastframe - firstframe, chunksize):
            yield snd.read(nframes)

    def getinfo(self, path: str) -> SndInfo:
        return self._getinfo(_soundfile.SoundFile(path, 'r'))

    def _getinfo(self, snd: _soundfile.SoundFile, ) -> SndInfo:
        encoding = _Soundfile.subtype_to_encoding[snd.subtype]
        fileformat = _Soundfile.format_to_fileformat[snd.format]
        metadata = {}
        for key in self.metadata_keys:
            value = getattr(snd, key, None)
            if value:
                metadata[key] = value

        ext = _os.path.splitext(snd.name)[1]
        if ext == '.ogg':
            metadata = util.tinytagMetadata(snd.name)
            bitrate = metadata.pop('bitrate', None)
        else:
            bitrate = None

        return SndInfo(samplerate=snd.samplerate,
                       nframes=snd.frames,
                       channels=snd.channels,
                       encoding=encoding,
                       fileformat=fileformat,
                       metadata=metadata,
                       bitrate=bitrate)

    @staticmethod
    def get_format_and_subtype(fmt: str, encoding: str = None
                               ) -> Tuple[str, str]:
        soundfile_fmt = {
            'wav': 'WAVEX',
            'aif': 'AIFF',
            'aiff': 'AIFF',
            'flac': 'FLAC',
        }.get(fmt)
        if not soundfile_fmt:
            raise ValueError(f"Format {fmt} not supported")
        subformat = _Soundfile.encoding_to_subtype.get(encoding)
        if not subformat:
            raise ValueError(f"Encoding {encoding} not supported for {fmt}")
        return soundfile_fmt, subformat


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
    # _writer = _PySndfileWriter

    def __init__(self, priority: int):

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
        import pysndfile
        self.pysndfile = pysndfile

    def read_with_info(self, path: str, start: float = 0, end: float = 0) -> Tuple[np.ndarray, SndInfo]:
        snd = self.pysndfile.PySndfile(path)
        info = self._getinfo(snd)
        samples = self._read(snd, start=start, end=end)
        return samples, info

    def _read(self, snd: pysndfile.PySndfile, start=0., end=0.) -> np.ndarray:
        sr: int = snd.samplerate()
        samp_start = int(start * sr)
        samp_end = int(end * sr) if end > 0 else snd.frames()
        if samp_start:
            snd.seek(samp_start)
        return snd.read_frames(samp_end - samp_start)

    def read_chunked(self, path: str, chunksize=2048, start: float = 0., stop: float = 0.
                     ) -> Iterator[np.ndarray]:
        snd = self.pysndfile.PySndfile(path)
        sr = snd.samplerate()
        if start:
            snd.seek(int(start*snd.samplerate()))
        firstframe = int(sr * start)
        lastframe = snd.frames() if stop == 0 else int(sr*stop)
        for pos, nframes in util.chunks(0, lastframe - firstframe, chunksize):
            yield snd.read_frames(nframes)

    def getinfo(self, path: str) -> SndInfo:
        return self._getinfo(self.pysndfile.PySndfile(path))

    def _getinfo(self, snd: pysndfile.PySndfile) -> SndInfo:
        metadataraw: Dict[str, bytes] = snd.get_strings()
        if metadataraw:
            metadata, extrainfo = {}, {}
            for k, v in metadataraw.items():
                if k not in self._keyTable:
                    extrainfo[k] = v
                else:
                    metadata[self._keyTable[k]] = v
        else:
            metadata, extrainfo = None, None

        return SndInfo(snd.samplerate(), snd.frames(), snd.channels(),
                       snd.encoding_str(), snd.major_format_str(),
                       metadata=metadata, extrainfo=extrainfo)

    def get_sndfile_format(self, fileformat: str, encoding: str) -> int:
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
        return self.pysndfile.construct_format(fileformat, f"{fmt}{bits}")

    def detect_format(self, path: str) -> Opt[str]:
        f = self.pysndfile.PySndfile(path)
        return self.pysndfile.fileformat_id_to_name.get(f.format())


# ----------------------------------------------------------------

class _Miniaudio(Backend):

    def __init__(self, priority):
        super().__init__(
                priority=priority,
                filetypes= ['mp3', 'ogg'],
                filetypes_write = [],
                can_read_chunked = True,
                can_write_chunked = False,
                name = 'miniaudio',
                supports_metadata=False
        )

    def getinfo(self, path: str) -> SndInfo:
        from . import backend_miniaudio
        ext = _os.path.splitext(path)[1]
        if ext == '.mp3':
            return backend_miniaudio.mp3info(path)
        elif ext == '.ogg':
            return backend_miniaudio.ogginfo(path)

    def read_with_info(self, path: str, start=0., end=0.) -> Tuple[np.ndarray, SndInfo]:
        return self.read(path, start, end)[0], self.getinfo(path)

    def read(self, path: str, start: float = 0., end: float = 0.) -> sample_t:
        from . import backend_miniaudio
        return backend_miniaudio.mp3read(path, start=start, end=end)

    def read_chunked(self, path: str, chunksize=2048, start: float = 0., stop: float = 0.
                     ) -> Iterator[np.ndarray]:
        from . import backend_miniaudio
        ext = _os.path.splitext(path)[1]
        if ext == '.mp3':
            return backend_miniaudio.mp3read_chunked(path, chunksize=chunksize, start=start, stop=stop)
        else:
            raise FormatNotSupported(f'chunked reading is not supported for {ext}')


_BACKENDS: Dict[str, Backend] = {
    'soundfile': _Soundfile(priority=0),
    'lameenc': _Lameenc(priority=10),
    'miniaudio': _Miniaudio(priority=10),
}

# if _is_package_installed('pysndfile'):
#     _BACKENDS['pysndfile'] = _PySndfile(priority=1)


_cache = {}


def report_backends():
    for b in _BACKENDS.values():
        if b.is_available():
            b.dump()
        else:
            print(f"Backend {b.name} NOT available")


def _get_backends() -> List[Backend]:
    backends = _cache.get('backends')
    if not backends:
        _cache['backends'] = backends = [b for b in _BACKENDS.values() if b.is_available()]
    return backends


def _get_backend(path: str = None, key: Callable[[Backend], bool] = None
                 ) -> Opt[Backend]:
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
    filetype = util.detect_format(path)
    backends = _get_backends()
    if key:
        backends = [b for b in backends if key(b)]
    if filetype:
        backends = [b for b in backends if filetype in b.filetypes]
    if backends:
        return min(backends, key=lambda backend: backend.priority)
    return None


def _get_write_backend(fileformat: str) -> Opt[Backend]:
    backends = _get_backends()
    if not backends:
        raise SndfileError("No available backends for writing")
    backends = [b for b in backends if fileformat in b.filetypes_write]
    if backends:
        return min(backends, key=lambda backend: backend.priority)
    return None
