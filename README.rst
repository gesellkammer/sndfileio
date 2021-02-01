SNDFILE.IO
==========

This package provides a simple and unified API to read and write sound-files to
and from numpy arrays. 

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


Dependencies
------------

-  numpy
-  scipy
-  ``pysndfile``

   -  https://pypi.python.org/pypi/pysndfile
   -  ``pip install pysndfile``
   -  extends support to many formats, like .flac, .ogg, etc.
   -  needs the sndfile library (``apt install libsndfile1-dev``)

-  ``python-samplerate``

   -  https://github.com/tuxu/python-samplerate
   -  Provides efficient and high quality resampling

-  ``miniaudio``

   -  https://github.com/irmen/pyminiaudio
   -  Provides support for .mp3 through a c extension, very efficient


License
-------

See the `LICENSE <LICENSE.md>`__ file for license rights and limitations
(MIT).
