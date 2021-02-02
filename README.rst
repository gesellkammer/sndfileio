*********
SNDFILEIO
*********

This package provides a simple and unified API to read and write sound-files to
and from numpy arrays. 

Documentation
-------------

https://sndfileio.readthedocs.io/en/latest/


API
---

sndread
~~~~~~~

-  reads all the samples (or a fragment) from a soundfile and returns a 
   tuplet (data, samplerate)
-  Data will always be as a ``numpy.float64``, between -1 and 1,
   independently of bit-rate

sndread_chunked
~~~~~~~~~~~~~~~

-  reads chunks of frames, avoiding the allocation of all the samples in
   memory

sndinfo
~~~~~~~

-  returns ``SndInfo``, a namedtuple with all the information of the
   sound-file

sndwrite
~~~~~~~~

-  writes the samples.
-  samples need to be a numpy.float64 array with data between -1 and 1

sndwrite_chunked
~~~~~~~~~~~~~~~~

-  allows you to write to the file as samples become available

resample
~~~~~~~~

Resample a numpy array to a new samplerate


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


Installation
------------

Make sure that you have ``libsamplerate`` installed.


.. code-block:: bash


    pip install sndfileio
    

Dependencies
------------

-  ``libsamplerate`` (``apt install libsndfile1-dev``)

All python dependencies are installed by ``pip``

License
-------

See the `LICENSE <LICENSE.md>`__ file for license rights and limitations
(MIT).
