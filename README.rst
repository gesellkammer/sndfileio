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
