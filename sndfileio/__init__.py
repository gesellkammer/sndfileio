"""
SNDFILE.IO

A simple module providing a unified API to read and write sound-files 
to and from numpy arrays. If no extra modules are installed, 
it uses the standard library to read and write uncompressed formats 
(WAV, AIFF)

If other modules are installed (scikits.audiolab, for example), it uses that
and more formats are supported

"""
from .sndfileio import *
from .resampling import resample

del resampling