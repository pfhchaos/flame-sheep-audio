"""
Build configuration for the optional C++ extension.

The extension is optional — if it fails to build (missing compiler, etc.),
the package installs without it and falls back to pure Python at runtime.
"""

import os
from setuptools import setup, Extension

# Allow skipping the build entirely (e.g., for CI without a C++ compiler)
if os.environ.get('FLAME_SHEEP_NO_NATIVE'):
    setup()
else:
    try:
        import pybind11
        import numpy as np

        native_dir = os.path.join('src', 'flame_sheep_audio', 'native')

        ext = Extension(
            'flame_sheep_audio._native',
            sources=[os.path.join(native_dir, 'octave_bank.cpp')],
            include_dirs=[
                pybind11.get_include(),
                np.get_include(),
                native_dir,  # for pocketfft_hdronly.h
            ],
            language='c++',
            extra_compile_args=[
                '-std=c++17',
                '-O3',
                '-march=native',
                '-ffast-math',
                '-fvisibility=hidden',
            ],
        )
        setup(ext_modules=[ext])
    except (ImportError, Exception):
        # pybind11 or numpy not available at build time — skip extension
        setup()
