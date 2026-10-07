# _prtcqt — pybind11 C++ extension for flame-sheep-audio.
#
# Compiles the vendored, header-only rt-cqt SlidingCqt (C++11/17) plus the
# pybind11 binding (prtcqt_bind.cpp) into an in-package extension module
# `flame_sheep_audio._prtcqt` (imported via `from . import _prtcqt` in
# _cqt_engine.py). The leading underscore marks it a private native module.
cmake_minimum_required(VERSION 3.15...3.30)

project(prtcqt LANGUAGES CXX)

set(RTCQT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/flame_sheep_audio/native/rt-cqt")

# scikit-build-core puts the build env's site-packages on CMAKE_PREFIX_PATH,
# so pybind11 (a build requirement) is found here.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_prtcqt
    "${RTCQT_DIR}/prtcqt_bind.cpp"
)

target_compile_features(_prtcqt PRIVATE cxx_std_17)
target_include_directories(_prtcqt PRIVATE "${RTCQT_DIR}")

# Install into the package so it imports as `flame_sheep_audio._prtcqt`.
install(TARGETS _prtcqt LIBRARY DESTINATION flame_sheep_audio)
