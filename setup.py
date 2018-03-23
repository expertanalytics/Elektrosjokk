# System imports
from distutils.core import setup
import platform
import sys
from os.path import join as pjoin

# Version number
major = 0
minor = 1


setup(
    name = "elektrosjokk",
    version = "{0}.{1}".format(major, minor),
    author = "Jakob E. Schreiner",
    author_email = "jakob@xal.no",
    packages = ["ect",],
)
