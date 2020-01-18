from __future__ import print_function
from setuptools import setup
import sys

version = "0.9.2"

print(version)

description = \
"""Common API for reading and writing soundfiles.   
* Uses installed packages if found (recommended: pysndfile, miniaudio)  
* Implements reading uncompressed formats correctly in any format.  
* The data is independent of the encoding. All data is presented as float64  
* Bitdepth is handled automatically depending on the the actual data  
"""

short_description = "Common API for reading writing soundfiles"

if sys.version_info < (3,6):
    sys.exit('Sorry, Python < 3.6 is not supported')

setup(
    name = "sndfileio",
    python_requires = ">=3.6",
    install_requires = [
        "numpy>1.8", 
        "scipy",
        "pysndfile",
        "miniaudio"],
    packages = ["sndfileio"],
    package_data = {'': ['README.md']},
    include_package_data = True,

    # Metadata
    version      = version,
    description  = short_description,
    summary      = "Simple API for reading and writing soundfiles", 
    author       = "Eduardo Moguillansky",
    author_email = "eduardo.moguillansky@gmail.com",
    url          = "https://github.com/gesellkammer/sndfileio",
    
    classifiers = [
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ]
)
