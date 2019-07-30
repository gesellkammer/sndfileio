# SNDFILE.IO

A simple module providing a unified API to read and write sound-files to and 
from numpy arrays. If no extra modules are installed, it uses only standard 
modules and numpy to read and write uncompressed formats (WAV, AIFF).

If other modules are installed (see supported backends), then they are used.

Even without third-party packages, it has certain advantages over the 
built-in modules wave and aifc.

* support for `PCM16`, `PCM24`, `PCM32` and `FLOAT32`
* unified output format, independent of encoding (always `float64`)
* unified API for all backends


## API

### sndread 

* reads ALL the samples and returns a tuplet (data, samplerate)
* Data will always be as a `numpy.float64`, between -1 and 1, independently of bit-rate

### sndread_chunked

* reads chunks of frames, avoiding the allocation of all the samples in memory

### sndinfo

* returns `SndInfo`, a namedtuple with all the information of the sound-file

### sndwrite

* writes the samples. 
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
   
* `pysndfile`
    * https://pypi.python.org/pypi/pysndfile
    * `pip install pysndfile`
    * extends support to many formats, like .flac, .ogg, etc.
    * needs the sndfile library (`apt install libsndfile1-dev`)

* `python-samplerate`
    * https://github.com/tuxu/python-samplerate
    * Provides efficient and high quality resampling

* `miniaudio`
	* https://github.com/irmen/pyminiaudio
	* Provides support for .mp3 through a c extension, very efficient

* `pydub`
    * https://github.com/jiaaro/pydub (`pip install pydub`)
    * Adds support for mp3
    * Depends on `ffmpeg` or `libav` being installed (`apt install ffmpeg`)
    * Not as efficient as miniaudio, depends on external binary being installed

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
