from __future__ import print_function
from setuptools import setup
import os

thisdir = os.path.abspath(os.path.dirname(__file__))
long_description = open(os.path.join(thisdir, 'README.rst')).read()

setup(
    version = "1.6.0",
    name = "sndfileio",
    python_requires = ">=3.8",
    install_requires = [
        "numpy>1.8", 
        "scipy",
        "pysndfile",
        "miniaudio",
        "nnresample",
        "numpyx>=0.4.1",
        "tinytag",
        "filetype",
        "lameenc"],
    packages = ["sndfileio"],
    package_data = {'': ['README.rst'], 'sndfileio': ['py.typed']},
    include_package_data = True,

    # Metadata
    description = "Simple API for reading / writing soundfiles",
    long_description = long_description,
    summary      = "Simple API for reading and writing soundfiles", 
    author       = "Eduardo Moguillansky",
    author_email = "eduardo.moguillansky@gmail.com",
    url          = "https://github.com/gesellkammer/sndfileio",
    license = 'LGPL',
)
