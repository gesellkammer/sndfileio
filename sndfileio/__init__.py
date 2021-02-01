"""
SNDFILE.IO

A simple module providing a unified API to read and write sound-files to and from
numpy arrays. If no extra modules are installed, it uses only standard modules
and numpy to read and write uncompressed formats (WAV, AIFF).

Backends
********

* PySndfile (supports wav, aif, flac, ogg, etc., https://pypi.org/project/pysndfile/)
* miniaudio (for mp3 support, https://pypi.org/project/miniaudio/)

API
****

* sndinfo(path): Returns a SndInfo, a namedtuple with all the information
    of the sound-file
* sndread(path): Reads ALL the samples. Returns a tuple (data, samplerate)
* sndwrite(samples, samplerate, outfile): Write samples to outfile
* sndwrite_like(samples, likefile, outfile): Write samples to outfile using
    likefile's parameters


Chunked IO
----------

* sndread_chunked(path): returns a generator yielding chunks of frames
* sndwrite_chunked(path): opens the file for writing. To write to the file,
    call .write on the returned handle

"""
from .sndfileio import *
from .resampling import resample
from .util import numchannels

del resampling