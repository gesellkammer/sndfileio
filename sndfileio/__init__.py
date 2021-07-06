"""

This package providesa unified API to read and write sound-files to and from
numpy arrays. It can read and write to `wav`, `aif`, `flac`, `mp3` and
all loss-less formats supported by `libsndfile`.

--------------------------------------------------

API
****

Read / write a file in one function.

:func:`sndinfo<sndfileio.sndinfo>`
==================================

    Returns a :class:`SndInfo` with all the information of the sound-file

:func:`sndread<sndfileio.sndread>`
==================================

    Reads ALL the samples. Returns a tuple (data, samplerate)

:func:`sndwrite<sndfileio.sndwrite>`
====================================

    Write samples to outfile

:func:`sndwrite_like<sndfileio.sndwrite_like>`
==============================================

    Write samples to outfile cloning another files format & encoding


:func:`sndget<sndfileio.sndget>`
==================================

    Reads the sample data, returns a tuple (**samples**, **info**), where
    `info` is a :class:`SndInfo` with all the information of the sound-file

-----------------------------


==========
Chunked IO
==========

It is possible to stream a soundfile by reading and processing chunks. This
is helpful in order to avoid allocating memory for a large. The same is possible
for writing

:func:`sndread_chunked<sndfileio.sndread_chunked>`
==================================================

    Returns a generator yielding chunks of frames

:func:`sndwrite_chunked<sndfileio.sndwrite_chunked>`
====================================================

    Opens the file for writing. Returns a :class:`SndWriter`. To write to the file,
    call :meth:`write` on the returned handle


---------


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
from . import util

del resampling
