"""
*********
SNDFILEIO
*********

A simple module providing a unified API to read and write sound-files to and from
numpy arrays. If no extra modules are installed, it uses only standard modules
and numpy to read and write uncompressed formats (WAV, AIFF).

Backends
********

* PySndfile (supports wav, aif, flac, ogg, etc., https://pypi.org/project/pysndfile/)
* miniaudio (for mp3 support, https://pypi.org/project/miniaudio/)

API
****

Read / write a file in one function.

:func:`sndinfo<sndfileio.sndinfo>`
----------------------------------

    Returns a :class:`SndInfo` with all the information of the sound-file

:func:`sndread<sndfileio.sndread>`
----------------------------------

    Reads ALL the samples. Returns a tuple (data, samplerate)

:func:`sndwrite<sndfileio.sndwrite>`
------------------------------------

    Write samples to outfile

:func:`sndwrite_like<sndfileio.sndwrite_like>`
----------------------------------------------

    Write samples to outfile cloning another files format & encoding


Chunked IO
----------

It is possible to stream a soundfile by reading and processing chunks. This
is helpful in order to avoid allocating memory for a large. The same is possible
for writing

:func:`sndread_chunked<sndfileio.sndead_chunked>`

    returns a generator yielding chunks of frames

:func:`sndwrite_chunked<sndfileio.sndwrite_chunked>`
    opens the file for writing. Returns a :class:`SndWriter`. To write to the file,
    call :meth:`write` on the returned handle


Examples
--------

.. code-block:: python

    # Normalize and save as flac
    from sndfileio import sndread, sndwrite
    samples, sr = sndread("in.wav")
    maxvalue = max(samples.max(), -samples.min())
    samples *= 1/maxvalue
    sndwrite(samples, sr, "out.flac")


.. code-block:: python

    # Process a file in chunks
    from sndfileio import *
    from sndfileio.dsp import
    with sndwrite_chunked(44100, "out.flac") as writer:
        for buf in sndread_chunked("in.flac"):
            # do some processing, like changing the gain
            buf *= 0.5
            writer.write(buf)


"""

from .sndfileio import *
from .resampling import resample
from .util import numchannels
from . import dsp

del resampling
