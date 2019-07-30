from __future__ import annotations
"""
SNDFILE.IO

A simple module providing a unified API to read and write sound-files to and from
numpy arrays. If no extra modules are installed, it uses only standard modules
and numpy to read and write uncompressed formats (WAV, AIFF).

If other modules are installed (scikits.audiolab, for example), it uses that
and more formats are supported.

Advantages over the built-in modules wave and aifc

* support for PCM16, PCM24, PCM32 and FLOAT32
* unified output format, independent of encoding --> always float64
* unified API for all backends

API
===

sndinfo(path)
    Returns a SndInfo, a namedtuple with all the information
    of the sound-file


Read & Write all samples
------------------------

sndread(path)
    Reads ALL the samples
    Returns a Sample - a tuplet (data, samplerate)
        data: a numpy.float64, normally between -1 and 1

sndwrite(samples, samplerate, outfile, encoding='auto')
    Write samples to outfile
    samples: a numpy.float64 array with data between -1 and 1

sndwrite_like(samples, likefile, outfile)
    Write samples to outfile using likefile's parameters
    (sr, format)

Chunked IO
----------

sndread_chunked(path)  --> returns a generator yielding chunks of frames

sndwrite_chunked(path) --> opens the file for writing.
                           To write to the file, call .write
"""
import os as _os
import struct as _struct
import warnings as _warnings
import numpy as np
import importlib
import logging

from .util import numchannels
from .datastructs import SndInfo, Sample
from typing import (
    Tuple, Union, Any, Iterator, Optional as Opt, 
    List, cast, IO, NamedTuple, Dict
)

logger = logging.getLogger("sndfileio")

__all__ = [
    "sndread",
    "sndread_chunked",
    "sndinfo",
    "sndwrite",
    "sndwrite_like",
    "sndwrite_chunked",
    "sndwrite_chunked_like",
    "bitdepth",
    "numchannels",
    "asmono",
    "getchannel"
]



class FormatNotSupported(Exception):
    pass


def _isPackageInstalled(pkg):
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


_CHUNKSIZE = 1024


########################################
#
#                API
#
########################################

class _SndWriter:
    
    def __init__(self, parent, sr:int, outfile:str, encoding:str) -> None:
        self.sr = sr
        self.outfile = outfile
        self.encoding = encoding
        self._parent = parent  
        self._file = None

    def write(self, frames: np.ndarray) -> None:
        pass

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None

    def __enter__(self) -> '_SndWriter':
        return self

    def __exit__(self) -> None:
        self.close()

    def get_backend(self):
        return self._parent.get_backend()

    @property
    def filetypes(self):
        return self._parent.filetypes_write


def sndread(path:str) -> Tuple[np.ndarray, int]:
    """
    Read a soundfile as a numpy array. This is a float array defined 
    between -1 and 1, independently of the format of the soundfile
    
    Returns (data:ndarray, sr:int)
    """
    backend = _getBackend(path)
    logger.debug(f"sndread: using backend {backend.name}")
    return backend.read(path)


def sndread_chunked(path:str, frames:int=_CHUNKSIZE) -> Iterator[np.ndarray]:
    """
    Returns a generator yielding numpy arrays (float64)
    of at most `frames` frames
    """
    backend = _getBackend(path, key=lambda backend: backend.can_read_chunked)
    if backend:
        logger.debug(f"sndread_chunked: using backend {backend.name}")
        return backend.read_chunked(path, frames)
    else:
        raise SndfileError("chunked reading is not supported by the available backends")


def sndinfo(path:str) -> SndInfo:
    """
    Get info about a soundfile

    path (str): the path to a soundfile

    RETURNS --> an instance of SndInfo: samplerate, nframes, channels, encoding, fileformat
    """
    backend = _getBackend(path)
    if not backend:
        logger.warn("sndinfo: no backend supports this filetype")
        return None
    logger.debug(f"sndinfo: using backend {backend.name}")
    return backend.getinfo(path)


def sndwrite(samples:np.ndarray, sr:int, outfile:str, encoding:str='auto') -> None:
    """
    samples  --> Array-like. the actual samples, shape=(nframes, channels)
    sr       --> Sampling-rate
    outfile  --> The name of the outfile. the extension will determine
                 the file-format.
                 The formats supported depend on the available backends
                 Without additional backends, only uncompressed formats
                 are supported (wav, aif)
    encoding --> one of:
                 - 'auto' or None: the encoding is determined from the format
                                   given by the extension of outfile, and
                                   from the data
                 - 'pcm16'
                 - 'pcm24'
                 - 'pcm32'
                 - 'flt32'

                 NB: not all file formats support all encodings.
                     Throws a SndfileError if the format does not support
                     the given encoding

          If set to 'auto', an encoding will be selected based on the
          file-format and on the data. The bitdepth of the data is
          measured, and if the file-format supports it, it will be used.
          For bitdepths of 8, 16 and 24 bits, a PCM encoding will be used.
          For a bitdepth of 32 bits, a FLOAT encoding will be used,
          or the next lower supported encoding
    """
    if encoding in ('auto', None):
        encoding = _guessEncoding(samples, outfile)
    # normalize in the case where there would be clipping
    clipping = ((samples > 1).any() or (samples < -1).any())
    if encoding.startswith('pcm') and clipping:
        maxvalue = max(samples.max(), abs(samples.min()))
        samples = samples / maxvalue
    backend = _getWriteBackend(outfile, encoding)
    if not backend:
        raise SndfileError("No backend found to support the given format")
    logger.debug(f"sndwrite: using backend {backend.name}")
    return backend.write(samples, sr, outfile, encoding)


def sndwrite_chunked(sr: int, outfile: str, encoding: str) -> _SndWriter:
    """
    Returns a SndWriter. Call its .write(samples) method 
    to write

    samples  --> Array-like. the actual samples, shape=(nframes, channels)
    sr       --> Sampling-rate
    outfile  --> The name of the outfile. the extension will determine
                 the file-format.
                 The formats supported depend on the available backends
                 Without additional backends, only uncompressed formats
                 are supported (wav, aif)
    encoding --> one of:
                 - 'pcm16'
                 - 'pcm24'
                 - 'pcm32'
                 - 'flt32'

    Example
    ~~~~~~~

    with sndwrite_chunked(44100, "out.flac", "pcm24") as writer:
        for buf in sndread_chunked("in.flac"):
            # do some processing, like changing the gain
            buf *= 0.5
            writer.write(buf)

    """
    backends = [backend for backend in _getBackends() if backend.can_write_chunked]
    if not backends:
        raise SndfileError("No backend found to support the given format")
    print(backends)
    backend = min(backends, key=lambda backend:backend.priority)
    logger.debug(f"sndwrite_chunked: using backend {backend.name}")
    return backend.writer(sr, outfile, encoding)


def asmono(samples:np.ndarray, channel:Union[int, str]=0) -> np.ndarray:
    """
    convert samples to mono if they are not mono already.

    The returned array will always have the shape (numframes,)

    channel: the channel number to use, or 'mix' to mix-down
             all channels
    """
    if numchannels(samples) == 1:
        # it could be [1,2,3,4,...], or [[1], [2], [3], [4], ...]
        if isinstance(samples[0], float):
            return samples
        elif isinstance(samples[0], np.dnarray):
            return np.reshape(samples, (len(samples),))
        else:
            raise TypeError("Samples should be numeric, found: %s"
                            % str(type(samples[0])))
    if isinstance(channel, int):
        return samples[:, channel]
    elif channel == 'mix':
        return _mix(samples, scale_by_numchannels=True)
    else:
        raise ValueError("channel has to be an integer indicating a channel,"
                         " or 'mix' to mix down all channels")


def getchannel(samples: np.ndarray, ch:int) -> np.ndarray:
    """
    Returns a view into a channel of samples.

    samples    : a numpy array representing the audio data
    ch         : the channel to extract (channels begin with 0)
    """
    N = numchannels(samples)
    if ch > (N - 1):
        raise ValueError("channel %d out of range" % ch)
    if N == 1:
        return samples
    return samples[:, ch]


def bitdepth(data:np.ndarray, snap:bool=True) -> int:
    """
    returns the number of bits actually used to represent the data.

    data: a numpy.array (mono or multi-channel)
    snap: snap to 8, 16, 24 or 32 bits.
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


def sndwrite_like(samples:np.ndarray, likefile:str, outfile:str) -> None:
    """
    Write samples to outfile with samplerate and encoding
    taken from likefile
    """
    info = sndinfo(likefile)
    sndwrite(samples, info.samplerate, outfile, encoding=info.encoding)


def sndwrite_chunked_like(likefile:str, outfile:str) -> _SndWriter:
    info = sndinfo(likefile)
    return sndwrite_chunked(info.samplerate, outfile, info.encoding)


############################################
#
#                BACKENDS
#
############################################


class _PySndfileWriter(_SndWriter):
    
    def _openFile(self, channels:int) -> None:
        major = _os.path.splitext(self.outfile)[1]
        if major not in self.filetypes:
            raise ValueError("Format %s not supported by this backend" % major)
        backend = self.get_backend()
        ext = _os.path.splitext(self.outfile)[1]
        sndformat = self._parent._getSndfileFormat(ext, self.encoding)
        self._file = backend.PySndfile(self.outfile, "w", sndformat, channels, self.sr)

    def write(self, frames:np.ndarray) -> None:
        if self._file:
            self._file.write_frames(frames)
        else:
            numchannels = frames.shape[1] if len(frames.shape) > 1 else 1
            self._openFile(numchannels)
            self.write(frames)

    def close(self):
        if self._file is None:
            raise IOError("Can't close, since this file was never open")
        self._file.writeSync()
        del self._file
        self._file = None


class _Backend:
    def __init__(self, priority, filetypes, filetypes_write, encodings, 
                 can_read_chunked, can_write_chunked, name):
        self.priority = priority
        self.filetypes = filetypes
        self.filetypes_write = filetypes_write
        self.encodings = encodings
        self.can_read_chunked = can_read_chunked
        self.can_write_chunked = can_write_chunked
        self.name = name
        self._backend = None
        self._writer = None

    def read(self):
        pass

    @staticmethod
    def _getBackend():
        return None

    def get_backend(self):
        if self._backend is not None:
            return self._backend
        self._backend = self._getBackend()
        return self._backend

    def is_available(self) -> bool:
        return _isPackageInstalled(self.name)
    
    def writer(self, sr:int, outfile:str, encoding:str) -> _SndWriter:
        if self._writer is None:
            raise SndfileError("This backend does not support writing")
        return self._writer(self, sr, outfile, encoding)

    def check_write(self, outfile, encoding):
        if encoding not in self.encodings:
            raise ValueError("Encoding not supported")
        ext = _os.path.splitext(outfile)[1].lower()
        if ext not in self.filetypes_write:
            raise ValueError(
                "The given format (%s) is not supported by the %s backend" %
                (ext, self.name)
            )

    def dump(self):
        print(f"Backend: {self.name} (available: {self.is_available}, priority: {self.priority})")
        if self.readtypes:
            readtypes = ", ".join(self.filetypes)
            print(f"    read types : {readtypes}")
        if self.filetypes_write:
            writetypes = ", ".join(self.filetypes_write)
            print(f"    write types: {writetypes}")
        ok, notok = "OK", "--"
        readchunked  = ok if self.can_read_chunked else notok
        writechunked = ok if self.can_write_chunked else notok
        print(f"    sndread_chunked: {readchunked}    sndwrite_chunked: {writechunked}")


class _PySndfile(_Backend):
    def __init__(self, priority):
        super().__init__(
            priority  = priority,
            filetypes = ".aif .aiff .wav .flac .ogg .wav64 .caf .raw".split(),
            filetypes_write = ".aif .aiff .wav .flac .ogg .wav64 .caf .raw".split(),
            encodings = 'pcm16 pcm24 flt32'.split(),
            can_read_chunked = True,
            can_write_chunked = True,
            name = 'pysndfile'
        )
        self._writer = _PySndfileWriter
    
    @staticmethod
    def _getBackend():
        try:
            import pysndfile
            return pysndfile
        except ImportError:
            return None
    
    def read(self, path:str) -> Sample:
        pysndfile = self.get_backend()
        snd = pysndfile.PySndfile(path)
        data = snd.read_frames(snd.frames())
        sr = snd.samplerate()
        return Sample(data, sr)

    def read_chunked(self, path:str, chunksize:int=_CHUNKSIZE
                     ) -> Iterator[np.ndarray]:
        pysndfile = self.get_backend()
        snd = pysndfile.PySndfile(path)
        for pos, nframes in _chunks(0, snd.frames(), chunksize):
            yield snd.read_frames(nframes)

    def getinfo(self, path:str) -> SndInfo:
        pysndfile = self.get_backend()
        snd = pysndfile.PySndfile(path)
        return SndInfo(snd.samplerate(), snd.frames(), snd.channels(),
                       snd.encoding_str(), snd.major_format_str())

    def write(self, data:np.ndarray, sr:int, outfile:str, encoding:str) -> None:
        self.check_write(outfile, encoding)
        ext = _os.path.splitext(outfile)[1].lower()
        fmt = self._getSndfileFormat(ext, encoding)
        pysndfile = self.get_backend()
        snd = pysndfile.PySndfile(outfile, mode='w', format=fmt,
                                  channels=numchannels(data), samplerate=sr)
        snd.write_frames(data)
        snd.writeSync()

    def _getSndfileFormat(self, extension, encoding):
        assert extension in self.filetypes
        fmt, bits = encoding[:3], int(encoding[3:])
        assert fmt in ('pcm', 'flt') and bits in (8, 16, 24, 32)
        extension = extension[1:]
        if extension == 'aif':
            extension = 'aiff'
        fmt = "%s%d" % (
            {'pcm': 'pcm', 
             'flt': 'float'}[fmt],
            bits
        )
        pysndfile = self.get_backend()
        return pysndfile.construct_format(extension, fmt)


class _AudiolabWriter(_SndWriter):
    
    def _openFile(self, channels:int) -> None:
        major = _os.path.splitext(self.outfile)[1][1:]
        if major not in self.filetypes:
            raise ValueError("Format %s not supported by this backend" % major)
        encoding = _normalizeEncoding(encoding)
        audiolab = self.get_backend()
        sndformat = audiolab.Format(major, encoding)
        self._file = audiolab.Sndfile(self.outfile, "w",
                                      sndformat, channels, self.sr)

    def write(self, frames:np.ndarray) -> None:
        if self._file:
            self._file.write_frames(frames)
        else:
            numchannels = frames.shape[1] if len(frames.shape) > 1 else 1
            self._openFile(numchannels)
            self.write(frames)


class _Audiolab(_Backend):
    def __init__(self, priority):
        filetypes = (".aif", ".aiff", ".wav", ".flac", ".ogg", ".wav64", ".caf", ".raw")
        super().__init__(
            name = 'scikits.audiolab',
            priority = priority,
            filetypes = filetypes,
            filetypes_write = filetypes,
            encodings = ('pcm16', 'pcm24', 'flt32'),
            can_read_chunked = True,
            can_write_chunked = True
        )
        self._writer = _AudiolabWriter

    def _getBackend(self):
        try:
            from scikits import audiolab
            return audiolab
        except ImportError:
            return None
    
    def read(self, path:str) -> Sample:
        audiolab = self.get_backend()
        snd = audiolab.Sndfile(path)
        data = snd.read_frames(snd.nframes)
        sr = snd.samplerate
        return Sample(data, sr)

    def read_chunked(self, 
                     path:str, 
                     chunksize:int=_CHUNKSIZE
                     ) -> Iterator[np.ndarray]:
        audiolab = self.get_backend()
        snd = audiolab.Sndfile(path)
        for pos, nframes in _chunks(0, snd.nframes, chunksize):
            yield snd.read_frames(nframes)

    def getinfo(self, path:str) -> SndInfo:
        audiolab = self.get_backend()
        snd = audiolab.Sndfile(path)
        return SndInfo(snd.samplerate, snd.nframes, snd.channels,
                       snd.encoding, snd.file_format)

    def write(self, data:np.ndarray, sr:int, outfile:str, encoding:str) -> None:
        self.check_write(outfile, encoding)
        ext = _os.path.splitext(outfile)[1]
        fmt = self._getSndfileFormat(ext, encoding)
        audiolab = self.get_backend()
        snd = audiolab.Sndfile(outfile, mode='w', format=fmt,
                               channels=numchannels(data), samplerate=sr)
        snd.write_frames(data)
        snd.close()

    def _getSndfileFormat(self, extension:str, encoding:str) -> Any:
        assert extension in self.filetypes
        fmt, bits = encoding[:3], int(encoding[3:])
        assert fmt in ('pcm', 'flt') and bits in (8, 16, 24, 32)
        extension = extension[1:]
        if extension == 'aif':
            extension = 'aiff'
        fmt2 = "%s%d" % (
            {'pcm': 'pcm', 'flt': 'float'}[fmt],
            bits
        )
        audiolab = self.get_backend()
        return audiolab.Format(extension, fmt2)
    
    
class _Builtin(_Backend):
    def __init__(self, priority):
        super().__init__(
            priority = priority,
            filetypes = ('.wav', '.aif', '.aiff'),
            filetypes_write = [],
            encodings = ('pcm16', 'pcm24', 'flt32'),
            can_read_chunked = True,
            can_write_chunked = False,
            name = 'builtin' 
        )
        self._writer = None

    def is_available(self):
        return True

    def read(self, path:str) -> np.ndarray:
        ext = _os.path.splitext(path)[1].lower()
        if ext in (".aif", ".aiff"):
            return _AiffReader(path).read()
        elif ext == ".wav":
            return _WavReader(path).read()
        else:
            raise ValueError("format not supported")

    def getinfo(self, path:str) -> SndInfo:
        ext = _os.path.splitext(path)[1].lower()
        if ext in (".aif", ".aiff"):
            return _AiffReader(path).getinfo()
        elif ext == ".wav":
            return _WavReader(path).getinfo()
        else:
            raise ValueError("Only sndfiles with ext. aif, aiff or wav are supported")

    def read_chunked(self, path:str, chunksize:int=_CHUNKSIZE) -> Iterator[np.ndarray]:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.wav':
            return _WavReader(path).read_chunked(chunksize)
        else:
            raise NotImplementedError("read_chunked not implemented")


class _PyDub(_Backend):
    def __init__(self, priority):
        super().__init__(
            priority=priority,
            filetypes = ('.mp3'),
            filetypes_write = [],
            encodings = ('pcm16',),
            can_read_chunked = False,
            can_write_chunked = False,
            name = 'pydub'
        )

    def is_available(self):
        return _isPackageInstalled("pydub")

    def getinfo(self, path:str) -> SndInfo:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            from . import mp3
            return mp3.mp3info(path)

    def read(self, path:str) -> np.ndarray:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            from . import mp3
            return mp3.mp3read(path)
        else:
            raise ValueError("format not supported by this backend")


class _Miniaudio(_Backend):

    def __init__(self, priority):
        super().__init__(
            priority=priority,
            filetypes= ['.mp3'],
            filetypes_write = [],
            can_read_chunked = False,
            can_write_chunked = False,
            encodings = ['pcm16'],
            name = 'miniaudio'
        )

    def is_available(self):
        return _isPackageInstalled("miniaudio")

    def getinfo(self, path:str) -> SndInfo:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            from . import backend_miniaudio
            return backend_miniaudio.mp3info(path)

    def read(self, path: str) -> np.ndarray:
        ext = _os.path.splitext(path)[1].lower()
        if ext == '.mp3':
            from . import backend_miniaudio
            return backend_miniaudio.mp3read(path)


BACKENDS = [
    _PySndfile(priority=0),
    _Miniaudio(priority=8), 
    # _PyDub(priority=10),
    _Builtin(priority=100), 
]  # type: List[_Backend]


def report_backends():
    for b in BACKENDS:
        if b.is_available():
            b.dump()
        else:
            print(f"Backend {b.name} NOT available")            
    

###########################################
#
#             IMPLEMENTATION
#
###########################################

class _WavReader:
    def __init__(self, path:str) -> None:
        self.path = path
        # fsize, self._bigendian = _wavReadRiff(open(path, "rb"))
        self._info = None  # type: Opt[SndInfo]

    def getinfo(self) -> SndInfo:
        if self._info is not None:
            return self._info
        self._info, extrainfo = _wavGetInfo(self.path)
        return self._info

    def read(self):
        sample, info = _wavRead(self.path)
        self._info = info
        return sample

    def read_chunked(self, chunksize:int=_CHUNKSIZE) -> Iterator[np.ndarray]:
        return _wavReadChunked(self.path, chunksize)


class _AiffReader:
    def __init__(self, path:str) -> None:
        self.path = path
        self._info = None  # type: Opt[SndInfo]

    def getinfo(self) -> SndInfo:
        if self._info is not None:
            return self._info
        self._info = _aifGetInfo(self.path)
        return self._info

    def read(self):
        sample, info = _aifRead(self.path)
        self._info = info
        return sample

                      
def _aifGetInfo(path:str) -> SndInfo:
    import aifc
    f = aifc.open(path)
    bytes = f.getsampwidth()
    if bytes == 4:
        raise IOError("32 bit aiff is not supported yet!")
    encoding = "pcm%d" % (bytes * 8)
    return SndInfo(f.getframerate(), f.getnframes(),
                   f.getnchannels(), encoding, "aiff")


def _aifRead(path:str) -> Tuple[Sample, SndInfo]:
    import aifc
    f = aifc.open(path)
    datastr = f.readframes(f.getnframes())
    bytes = f.getsampwidth()
    channels = f.getnchannels()
    encoding = "pcm%d" % (bytes * 8)
    if encoding == 'pcm8':
        data = (np.fromstring(datastr, dtype=np.int8)/(2.0 ** 7)).astype(float)
    elif encoding == 'pcm16':
        data = (np.fromstring(datastr, dtype=">i2")/(2.0 ** 15)).astype(float)
    elif encoding == 'pcm24':
        data = np.fromstring(datastr, dtype=np.ubyte)
        data = _numpy24to32bit(data, bigendian=True).astype(float)/(2.0 ** 31)
    elif encoding == 'pcm32':
        data = (np.fromstring(datastr, dtype=">i4")/(2.0 ** 31)).astype(float)
    if channels > 1:
        data = data.reshape(-1, channels)
    info = SndInfo(f.getframerate(), f.getnframes(),
                   f.getnchannels(), encoding, "aiff")
    return Sample(data, info.samplerate), info
        

def _wavReadRiff(fid) -> Tuple[int, bool]:
    bigendian = False
    asbytes = lambda x: bytes(x, "ascii")
    str1 = fid.read(4)
    if str1 == asbytes('RIFX'):
        bigendian = True
    elif str1 != asbytes('RIFF'):
        raise ValueError("Not a WAV file.")
    if bigendian:
        fmt = '>I'
    else:
        fmt = '<I'
    fsize = _struct.unpack(fmt, fid.read(4))[0] + 8
    str2 = fid.read(4)
    if (str2 != asbytes('WAVE')):
        raise ValueError("Not a WAV file.")
    if str1 == asbytes('RIFX'):
        bigendian = True
    return fsize, bigendian


def _wavReadFmt(f, bigendian:bool) -> Tuple[int, str, int, int, int, int, int]:
    fmt = b">" if bigendian else b"<"
    res = _struct.unpack(fmt + b'ihHIIHH', f.read(20))  
    chunksize, format, ch, sr, brate, ba, bits = res
    formatstr = {
        1: 'pcm',
        3: 'flt',
        6: 'alw',
        7: 'mlw',
       -2: 'ext'  # extensible
    }.get(format)
    if formatstr is None:
        raise SndfileError("could not understand format while reading")
    if formatstr == 'ext':
        raise SndfileError("extension formats are not supported yet")
    if chunksize > 16:
        f.read(chunksize - 16)
    return chunksize, formatstr, ch, sr, brate, ba, bits


def _wavReadData(fid, 
                 size:int, 
                 channels:int, 
                 encoding:str, 
                 bigendian:bool) -> np.ndarray:
    """
    adapted from scipy.io.wavfile._read_data_chunk

    assume we are at the data (after having read the size)
    """
    bits = int(encoding[3:])
    if bits == 8:
        data = np.fromfile(fid, dtype=np.ubyte, count=size)
        if channels > 1:
            data = data.reshape(-1, channels)
    else:
        bytes = bits // 8
        if encoding in ('pcm16', 'pcm32', 'pcm64'):
            if bigendian:
                dtype = '>i%d' % bytes
            else:
                dtype = '<i%d' % bytes
            data = np.fromfile(fid, dtype=dtype, count=size // bytes)
            if channels > 1:
                data = data.reshape(-1, channels)
        elif encoding[:3] == 'flt':
            print("flt32!")
            if bits == 32:
                if bigendian:
                    dtype = '>f4'
                else:
                    dtype = '<f4'
            else:
                raise NotImplementedError
            data = np.fromfile(fid, dtype=dtype, count=size // bytes)
            if channels > 1:
                data = data.reshape(-1, channels)
        elif encoding == 'pcm24':
            # this conversion approach is really bad for long files
            # TODO: do the same but in chunks
            data = _numpy24to32bit(np.fromfile(fid, dtype=np.ubyte, count=size), 
                                   bigendian=False)
            if channels > 1:
                data = data.reshape(-1, channels)
    return data


def _wavRead(path:str, asfloat:bool=True) -> Tuple[Sample, SndInfo]:
    with open(path, 'rb') as f:
        info, extrainfo = _wavGetInfo(f)
        data = _wavReadData(f, extrainfo['datasize'], info.channels, info.encoding, 
                            extrainfo['bigendian'])
    if asfloat:
        data = _floatize(data, info.encoding).astype(float)
    return Sample(data, info.samplerate), info


def _wavReadChunked(path:str, frames:int=100, asfloat:bool=True
                    ) -> Iterator[np.ndarray]:
    with open(path, 'rb') as f:
        info, extrainfo = _wavGetInfo(f)
        if info.encoding == 'flt32':
            raise NotImplementedError("float32 is not correctly implemented")
        bits = int(info.encoding[3:])
        bytes = bits // 8
        chunksize = bytes * info.channels * frames
        if bits == 8:
            raise NotImplementedError("8 bit .wav is not supported")
        dtype = ('>i%d' if extrainfo['bigendian'] else '<i%d') % bytes
        for _, chunk in _chunks(0, chunksize, extrainfo['datasize']):
            data = np.fromfile(f, dtype=dtype, count=chunk // bytes)
            if info.channels > 1:
                data = data.reshape(-1, info.channels)
            if asfloat:
                data = _floatize(data, info.encoding)
            yield data


def _wavGetInfo(f:Union[IO, str]) -> Tuple[SndInfo, Dict[str, Any]]:
    """
    Read the info of a wav file. taken mostly from scipy.io.wavfile

    if extended: returns also fsize and bigendian
    """
    if isinstance(f, (str, bytes)):
        f = open(f, 'rb')
        needsclosing = True
    else:
        needsclosing = False
    fsize, bigendian = _wavReadRiff(f)
    fmt = ">i" if bigendian else "<i"
    while (f.tell() < fsize):
        chunk_id = f.read(4)
        if chunk_id == b'fmt ':
            chunksize, sampfmt, chans, sr, byterate, align, bits = _wavReadFmt(f, bigendian)
        elif chunk_id == b'data':
            datasize = _struct.unpack(fmt, f.read(4))[0]
            nframes = int(datasize / (chans * (bits / 8)))
            break
        else:
            _warnings.warn("chunk not understood: %s" % chunk_id)
            data = f.read(4)
            size = _struct.unpack(fmt, data)[0]
            f.seek(size, 1)
    encoding = _encoding(sampfmt, bits)
    if needsclosing:
        f.close()
    info = SndInfo(sr, nframes, chans, encoding, "wav")
    return info, {'fsize': fsize, 'bigendian': bigendian, 'datasize': datasize}
    

#   HELPERS ------------------------------------


def _floatize(data:np.ndarray, encoding:str) -> np.ndarray:
    assert (data > 0).any()
    if encoding == 'flt32':
        return data
    elif encoding == 'pcm24':
        return data / (2.0 ** 31)
    elif encoding == 'pcm16':
        return data / (2.0 ** 15)
    else:
        raise ValueError("encoding not understood")


def _encoding(format:str, bits:int) -> str:
    """
    format, bits as returned by _wavReadFmt

    format: "pcm", "float"
    bits  : 16, 24, 32
    """
    return "%s%d" % (format, bits)


def _normalizeEncoding(encoding:Union[int, str]) -> str:
    if isinstance(encoding, int):
        if encoding in (16, 24):
            return 'pcm%d' % encoding
        elif encoding in (32, 64):
            return 'flt32'
        else:
            raise ValueError("encoding not supported")
    encoding = encoding.lower()
    if encoding in ('flt32', 'float', 'float32'):
        return 'flt32'
    elif encoding in ('flt64', 'float64', 'double'):
        return 'flt64'
    return encoding


def _numpy24to32bit(data:np.ndarray, bigendian:bool=False) -> np.ndarray:
    """
    data is a ubyte array of shape = (size,) 
    (interleaved channels if multichannel)
    """
    target = np.zeros((data.shape[0] * 4 / 3,), dtype=np.ubyte)
    if not bigendian:
        target[3::4] = data[2::3]
        target[2::4] = data[1::3]
        target[1::4] = data[0::3]
    else:
        target[1::4] = data[2::3]
        target[2::4] = data[1::3]
        target[3::4] = data[0::3]
    del data
    targetraw = target.tostring()
    del target
    data = np.fromstring(targetraw, dtype=np.int32)
    return data


def _getBackends():
    return [b for b in BACKENDS if b.is_available()]
  

def _getBackend(path=None, key=None):
    """
    key: a function (backend -> bool) signaling if the backend 
         is suitable for a specific task

    example
    =======

    # Get available backends which can read in chunks
    >>> backend = _getBackend('file.flac', key=lambda backend:backend.can_read_chunked())
    """
    ext = _os.path.splitext(path)[1].lower() if path else None
    backends = _getBackends()
    if key:
        backends = [b for b in backends if key(b)]
    if ext:
        backends = [b for b in backends if ext in b.filetypes]
    if backends:
        return min(backends, key=lambda backend: backend.priority)
    return None


def _getWriteBackend(outfile:str, encoding:str):
    backends = _getBackends()
    if not backends:
        raise SndfileError("No available backends for writing")
    ext = _os.path.splitext(outfile)[1].lower()
    if ext:
        backends = [b for b in backends if ext in b.filetypes_write]
    if backends:
        return min(backends, key=lambda backend: backend.priority)
    return None


def _mix(samples:np.ndarray, scale_by_numchannels:bool=True) -> np.ndarray:
    summed = samples.sum(0)
    if scale_by_numchannels:
        summed *= (1 / numchannels(samples))
    return summed


def _guessEncoding(data:np.ndarray, outfile:str) -> str:
    ext = _os.path.splitext(outfile)[1].lower()
    maxbits = min(32, bitdepth(data, snap=True))
    if ext in ('.wav', '.aif', '.aiff'):
        encoding = {
            16: 'pcm16',
            24: 'pcm24',
            32: 'flt32',
        }.get(maxbits, 'flt32')
    elif ext == ".flac":
        encoding = {
            16: 'pcm16',
            24: 'pcm24',
            32: 'pcm24'
        }.get(maxbits, 'pcm24')
    else:
        raise FormatNotSupported(f"The format {ext} is not supported")
    assert encoding in ('pcm16', 'pcm24', 'flt32')
    return encoding


del Tuple, Union, Any, Iterator, Opt, List, cast, IO, NamedTuple
