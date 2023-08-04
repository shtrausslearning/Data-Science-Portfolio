from setuptools import find_packages, setup

import numpy as np
from Cython.Build import cythonize

setup(
    name='mlmodels',
    version='1.0',
    packages=['.mlmodels'],
)