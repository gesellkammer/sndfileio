# SNDFILE.IO

A simple module providing a unified API to read and write sound-files to and from numpy arrays. If no extra modules are installed, it uses only standard modules and numpy to read and write uncompressed formats (WAV, AIFF).

If other modules are installed (scikits.audiolab, for example), then they are used.

Even without third-party packages, it has certain advantages over the built-in modules wave and aifc

* support for `PCM16`, `PCM24`, `PCM32` and `FLOAT32`
* unified output format, independent of encoding (always `float64`)
* unified API for all backends

## API

### sndread 

* it will read ALL the samples and return a Sample (a tuplet data, samplerate)
* Data will always be as a `numpy.float64`, between -1 and 1, independently of bit-rate

### sndread_chunked

* will read chunks of frames, avoiding the allocation of all the samples in memory

### sndinfo

* return `SndInfo`, a namedtuple with all the information of the sound-file

### sndwrite

* write the samples. 
* samples need to be a numpy.float64 array with data between -1 and 1

### sndwrite_chunked

* allows you to write to the file as samples become available

### resample

Resample a numpy array to a new samplerate

## Dependencies

### Mandatory
 
   * numpy
   * scipy

### Optional Backends
   
   * scikits.audiolab
   * PySndfile: https://pypi.python.org/pypi/pysndfile
   * python-samplerate: https://github.com/tuxu/python-samplerate
   
