from distutils.core import setup
from setuptools import Extension
import numpy as np
import os

is_debug = int(os.environ.get('DEBUG', '0')) == 1
is_profile = int(os.environ.get('PROFILE', '0')) == 1

extra_args = ['-std=c11', '-Wall', '-march=native', '-fopenmp']
libraries = []

if is_debug:
    extra_args += ['-O0', '-g3', '-DDEBUG']

else:
    extra_args += ['-O3']

if is_profile:
    extra_args += ['-lprofiler']
    if '-O3' in extra_args:
        extra_args.remove('-O3')

    if '-O0' not in extra_args:
        extra_args +=['-O0']

    if '-g3' not in extra_args:
        extra_args +=['-g3']

extension = Extension('reversi.traverse.traverse',
                      include_dirs=['./reversi/traverse/include'],
                      libraries=libraries,
                      library_dirs=[],
                      extra_compile_args=extra_args,
                      extra_link_args=extra_args,
                      sources=['./reversi/traverse/src/traverse.c'],
                      language='c')

setup(
    name='sparse_convolution',
    ext_modules=[extension],
    include_dirs=[
        np.get_include()
    ],
    requires=[
        'Cython',
        'numpy'
    ]
)
