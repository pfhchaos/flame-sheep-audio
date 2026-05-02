"""
Build script for the native C++ extension.

Usage (development):
    cd flame_sheep_audio/src/flame_sheep_audio/native
    python setup_native.py build_ext --inplace

The resulting _native.cpython-*.so gets placed next to the Python package.
"""

from setuptools import setup, Extension
import pybind11
import numpy as np

ext = Extension(
    'flame_sheep_audio._native',
    sources=['octave_bank.cpp'],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
        '.',  # for pocketfft_hdronly.h
    ],
    language='c++',
    extra_compile_args=[
        '-std=c++17',
        '-O3',           # full optimization
        '-march=native', # use CPU-specific instructions (AVX2, etc.)
        '-ffast-math',   # allow FP reordering for speed
        '-fvisibility=hidden',  # smaller .so
    ],
)

setup(
    name='flame_sheep_audio._native',
    ext_modules=[ext],
)
