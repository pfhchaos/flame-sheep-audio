/*
 * prtcqt — Python bindings for rt-cqt SlidingCqt.
 *
 * Only exposes SlidingCqt12 and SlidingCqt24 (real-time streaming).
 * The non-sliding ConstantQTransform is not used.
 *
 * Original: Jonas Merkt (BSD-3), vendored and trimmed for flame-sheep.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "Python_SlidingCqt.h"

namespace py = pybind11;

static constexpr bool use_windowing{true};

PYBIND11_MODULE(prtcqt, m)
{
    m.doc() = "Real-time Constant Q Transform (SlidingCqt)";

    py::class_<Cqt::Python_SlidingCqt<12, 9, use_windowing>>(m, "SlidingCqt12")
        .def(py::init<>())
        .def("init", &Cqt::Python_SlidingCqt<12, 9, use_windowing>::init)
        .def("inputBlock", &Cqt::Python_SlidingCqt<12, 9, use_windowing>::Python_inputBlock)
        .def("outputBlock", &Cqt::Python_SlidingCqt<12, 9, use_windowing>::Python_outputBlock)
        .def("getOctaveValues", &Cqt::Python_SlidingCqt<12, 9, use_windowing>::Python_getOctaveValues)
        .def("getOctaveBinFreqs", &Cqt::Python_SlidingCqt<12, 9, use_windowing>::Python_getOctaveBinFreqs);

    py::class_<Cqt::Python_SlidingCqt<24, 9, use_windowing>>(m, "SlidingCqt24")
        .def(py::init<>())
        .def("init", &Cqt::Python_SlidingCqt<24, 9, use_windowing>::init)
        .def("inputBlock", &Cqt::Python_SlidingCqt<24, 9, use_windowing>::Python_inputBlock)
        .def("outputBlock", &Cqt::Python_SlidingCqt<24, 9, use_windowing>::Python_outputBlock)
        .def("getOctaveValues", &Cqt::Python_SlidingCqt<24, 9, use_windowing>::Python_getOctaveValues)
        .def("getOctaveBinFreqs", &Cqt::Python_SlidingCqt<24, 9, use_windowing>::Python_getOctaveBinFreqs);

    m.attr("__version__") = "dev";
}
