from __future__ import print_function
from setuptools import setup
import sys, os

version = "1.2.0"


short_description = "Common API for reading writing soundfiles"

if sys.version_info < (3,8):
    sys.exit('Sorry, Python < 3.8 is not supported')

# read the contents of your README file
thisdir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(thisdir, 'README.rst')) as f:
    long_description = f.read()


setup(
    name = "sndfileio",
    python_requires = ">=3.8",
    install_requires = [
        "numpy>1.8", 
        "scipy",
        "pysndfile",
        "miniaudio",
        "nnresample",
        "numpyx>=0.4.1"],
    packages = ["sndfileio"],
    package_data = {'': ['README.rst']},
    include_package_data = True,

    # Metadata
    version      = version,
    description = short_description,
    long_description = long_description,
    summary      = "Simple API for reading and writing soundfiles", 
    author       = "Eduardo Moguillansky",
    author_email = "eduardo.moguillansky@gmail.com",
    url          = "https://github.com/gesellkammer/sndfileio",
    license = 'LGPL',
    
    classifiers = [
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ]
)
