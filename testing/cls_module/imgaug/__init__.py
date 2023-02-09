"""Imports for package imgaug."""
from __future__ import absolute_import

# this contains some deprecated classes/functions pointing to the new
# classes/functions, hence always place the other imports below this so that
# the deprecated stuff gets overwritten as much as possible
from .imgaug import *  # pylint: disable=redefined-builtin

from . import augmentables as augmentables
from .augmentables import *
from . import augmenters as augmenters
from . import parameters as parameters
from . import dtypes as dtypes
from . import data as data

__version__ = '0.4.0'
