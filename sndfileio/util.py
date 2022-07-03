from __future__ import annotations
import numpy as np
import os


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Optional as Opt, Iterator, Tuple, Union



encodings_for_format = {
    'wav': ['pcm16', 'pcm24', 'pcm32', 'float32', 'float64'],
    'aiff': ['pcm16', 'pcm24', 'pcm32', 'float32', 'float64'],
    'flac': ['pcm16', 'pcm24'],
    'mp3': ['pcm16', 'pcm24'],
    'ogg': ['pcm16']
}

_default_encoding = {
    'wav': 'float32',
    'aif': 'float32',
    'aiff': 'float32',
    'mp3': 'pcm16',
    'flac': 'pcm24',
    'ogg': 'pcm16'
}


metadata_possible_keys = {'comment', 'title', 'artist', 'album', 'tracknumber', 'software'}


known_fileformats = encodings_for_format.keys()


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
        >>> from sndfileio import *
        >>> stereosamples, sr = sndread("stereo.wav")
        >>> ch0 = util.getchannel(stereosamples, 0)
        >>> ch0 *= 0.5
        >>> sndwrite_like("out.wav", stereosamples, likefile="stereo.wav")

    """
    N = numchannels(samples)
    if channel > (N - 1):
        raise ValueError(f"channel {channel} out of range (max. {N} channels)")
    if N == 1:
        return samples
    return samples[:, channel]


def numchannels(samples:np.ndarray) -> int:
    """
    return the number of channels present in samples

    Args:
        samples: a numpy array as returned by sndread

    Returns:
        the number of channels in `samples`

    For multichannel audio, samples is always interleaved,
    meaning that samples[n] returns always a frame, which
    is either a single scalar for mono audio, or an array
    for multichannel audio.
    """
    if len(samples.shape) == 1:
        return 1
    else:
        return samples.shape[1]


def apply_multichannel(data:np.ndarray, func:Callable[[np.ndarray], np.ndarray]
                       ) -> np.ndarray:
    """
    Apply a function ``(samples1D) -> samples1D`` along the channels of a multichannel
    sample array, returning a multichannel sample array of the same shape

    Args:
        data: the samples
        func: the function to apply to each channel

    Returns:
        the resulting samples

    Example::

        >>> import numpy as np
        >>> source = np.array([[0.,  0.],
        ...                    [0.1, 0.2],
        ...                    [0.3, 0.4.],
        ...                    [0.5, 0.6]])
        >>> apply_multichannel(source, lambda chan: chan*0.5)
        array([[0.,   0. ]
               [0.05, 0.1]
               [0.15, 0.2]
               [0.25, 0.3]])
    """
    ch = numchannels(data)
    if ch == 1:
        return func(data)
    else:
        chans = []
        for i in range(ch):
            chans.append(func(data[:,i]))
        out = np.concatenate(chans)
        out.shape = (ch, len(data))
        return out.T


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


def mix(samples:np.ndarray, scale_by_numchannels:bool=True) -> np.ndarray:
    summed = samples.sum(0)
    if scale_by_numchannels:
        summed *= (1 / numchannels(samples))
    return summed


def fileformat_from_ext(ext: str) -> Opt[str]:
    """
    Deduces the file format from the extension

    The returned file format is always a lowercase string,
    one of "wav", "aiff", "flac", "ogg", "wv", "mp3".
    Reurns None if not a supported extension

    Examples
    --------

    >>> fileformat_from_ext(("out.wav"))
    wav
    >>> fileformat_from_ext(("foo.FLAC"))
    flac
    >>> fileformat_from_ext(("bar.aif"))
    aiff

    """
    if ext.startswith("."):
        ext = ext[1:]
    ext = ext.lower()
    if ext == "aif":
        return "aiff"
    return ext if ext in known_fileformats else None


def detect_format(path:str) -> Opt[str]:
    """
    Detect the file format of a given soundfile

    Args:
        path: the path to the soundfile

    Returns:
        the fileformat, or None if not a known file format
    """
    ext = os.path.splitext(path)[1][1:].lower() if path else None
    if ext in known_fileformats:
        return ext
    import filetype
    ext = filetype.guess_extension(path)
    return ext if ext in known_fileformats else None


def default_encoding(fileformat:str) -> Opt[str]:
    """
    Return the default encoding for the given fileformat

    =======    =================
    Format     Default encoding
    =======    =================
     wav        float32
     aif        float32
     flac       pcm24
     mp3        pcm16
    =======    =================

    """
    return _default_encoding.get(fileformat)


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
        return mix(samples, scale_by_numchannels=True)
    else:
        raise ValueError("channel has to be an integer indicating a channel,"
                         " or 'mix' to mix down all channels")


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


def samples_out_of_range(data:np.ndarray) -> bool:
    """
    Returns True if any sample is out of the range -1:1
    """
    import numpyx
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
        if samples_out_of_range(data):
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
    elif fmt == 'mp3':
        encoding = 'pcm16'
    else:
        raise ValueError(f"The format {fmt} is not supported")
    assert encoding in ('pcm16', 'pcm24', 'float32')
    return encoding


def chunks(start:int, end:int, step:int) -> Iterator[Tuple[int, int]]:
    pos = start
    last_full = end - step
    while pos < last_full:
        yield pos, step
        pos += step
    yield pos, end - pos


def tinytagMetadata(path: str) -> dict:
    import tinytag
    m = tinytag.TinyTag.get(path)
    metadata = {}
    if m.title:
        metadata['title'] = m.title
    if m.album:
        metadata['album'] = m.album
    if m.comment:
        metadata['comment'] = m.comment
    if m.artist:
        metadata['artist'] = m.artist
    if m.track:
        metadata['tracknumber'] = m.track
    if m.bitrate:
        metadata['bitrate'] = m.bitrate
    return metadata
