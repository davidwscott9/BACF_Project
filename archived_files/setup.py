# Adapted from https://github.com/lawpdas/fhog-python/blob/master/python3/fhog/setup.py
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('gradientMex', sources = ['gradientMex.cpp']) ]

# setup(
#         name = 'python MATLAB fhog',
#         version = '1.0',
#         include_dirs = [np.get_include()], #Add Include path of numpy
#         ext_modules = ext_modules
#       )

setup(
    name= 'Generic model class',
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = ext_modules)
