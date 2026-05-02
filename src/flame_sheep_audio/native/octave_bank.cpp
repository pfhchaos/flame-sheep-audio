/*
 * octave_bank.cpp — C++ implementation of OctaveBankEngine.push_hop
 *
 * Replaces the Python loop over 9 octaves with a single C++ call.
 * Each octave: slide buffer, window, FFT, magnitude, mask, RMS-rebin.
 *
 * Uses pocketfft (header-only, same library numpy uses internally).
 * Exposes a stateful OctaveBankNative class to Python via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// pocketfft — header-only FFT library
// We only need real-to-complex forward transforms.
#include "pocketfft_hdronly.h"

namespace py = pybind11;


// ---------------------------------------------------------------------------
// Per-octave configuration — mirrors the Python _build_octave_bank() output.
// Built once at construction, immutable after that.
// ---------------------------------------------------------------------------
struct OctaveConfig {
    int fft_size;
    int n_fft_bins;      // number of FFT bins in this octave's frequency range
    float f_lo, f_hi;

    // Precomputed Hann window (length = fft_size)
    std::vector<float> window;

    // Boolean mask over rfft output: which bins fall in [f_lo, f_hi)
    // Stored as indices for direct gather (faster than boolean mask).
    std::vector<int> mask_indices;

    // Sliding sample buffer (length = fft_size)
    std::vector<float> buffer;
};


// ---------------------------------------------------------------------------
// OctaveBankNative — the C++ engine.
//
// Construction mirrors Python's _build_octave_bank() + OctaveBankEngine.__init__.
// push_hop() mirrors Python's push_hop() but does everything in one call.
// ---------------------------------------------------------------------------
class OctaveBankNative {
public:
    /*
     * Constructor: build the octave bank configuration.
     *
     * Same parameters as the Python version:
     *   sr              — sample rate (default 48000)
     *   hop             — hop size (default 512)
     *   n_octaves       — number of octaves (default 9)
     *   bins_per_octave — output bins per octave (default 12)
     *   fmin            — lowest frequency (default 32.7 Hz, C1)
     */
    OctaveBankNative(int sr, int hop, int n_octaves, int bins_per_octave, float fmin)
        : sr_(sr), hop_(hop), n_octaves_(n_octaves), bpo_(bins_per_octave), fmin_(fmin)
    {
        n_bins_ = n_octaves * bins_per_octave;

        // Build per-octave configs — same logic as Python's _build_octave_bank()
        for (int oct = 0; oct < n_octaves; ++oct) {
            OctaveConfig cfg;
            cfg.f_lo = fmin * std::pow(2.0f, oct);
            cfg.f_hi = fmin * std::pow(2.0f, oct + 1);
            float octave_width = cfg.f_hi - cfg.f_lo;

            // FFT size: need enough bins to get bins_per_octave in this range
            // bin_width = sr / fft_size
            // octave_width / bin_width >= bins_per_octave
            // fft_size >= sr * bins_per_octave / octave_width
            int needed = static_cast<int>(
                std::ceil(static_cast<float>(sr) * bins_per_octave / octave_width));

            // Round up to power of 2
            int fft_size = std::max(hop, 1 << static_cast<int>(
                std::ceil(std::log2(std::max(needed, hop)))));
            fft_size = std::min(fft_size, 16384);  // cap for latency

            cfg.fft_size = fft_size;

            // Build Hann window (periodic, same as scipy's sym=False)
            // w[n] = 0.5 * (1 - cos(2*pi*n / N))
            cfg.window.resize(fft_size);
            for (int i = 0; i < fft_size; ++i) {
                cfg.window[i] = 0.5f * (1.0f - std::cos(
                    2.0f * static_cast<float>(M_PI) * i / fft_size));
            }

            // Compute which rfft bins fall in [f_lo, f_hi)
            // rfft produces fft_size/2 + 1 bins
            int n_rfft = fft_size / 2 + 1;
            cfg.mask_indices.clear();
            for (int k = 0; k < n_rfft; ++k) {
                float freq = static_cast<float>(k) * sr / fft_size;
                if (freq >= cfg.f_lo && freq < cfg.f_hi) {
                    cfg.mask_indices.push_back(k);
                }
            }
            cfg.n_fft_bins = static_cast<int>(cfg.mask_indices.size());

            // Zero-init the sliding buffer
            cfg.buffer.assign(fft_size, 0.0f);

            octaves_.push_back(std::move(cfg));
        }

        // Previous spectrum for flux computation (starts as zeros)
        prev_spectrum_.assign(n_bins_, 0.0f);
        has_prev_ = false;

        // Pre-allocate scratch buffers for the largest FFT size.
        // These get reused every call instead of allocating on the heap.
        int max_fft = 0;
        for (auto& o : octaves_) max_fft = std::max(max_fft, o.fft_size);
        scratch_windowed_.resize(max_fft);
        scratch_spectrum_.resize(max_fft / 2 + 1);
        // Masked bins scratch — sized for max possible masked bins
        int max_masked = 0;
        for (auto& o : octaves_) max_masked = std::max(max_masked, o.n_fft_bins);
        scratch_masked_.resize(max_masked);
    }

    /*
     * push_hop — the hot path.
     *
     * Takes a numpy array of `hop` float32 samples.
     * Returns a tuple: (magnitude, flux, zcr)
     *   magnitude: float32[n_bins]  — RMS-rebinned log-spaced spectrum
     *   flux:      float32[n_bins]  — half-wave rectified difference from previous
     *   zcr:       float            — zero-crossing rate of this hop
     *
     * The waveform is NOT returned — Python keeps its own copy of the hop.
     * This avoids an unnecessary memcpy.
     */
    std::tuple<py::array_t<float>, py::array_t<float>, float>
    push_hop(py::array_t<float, py::array::c_style | py::array::forcecast> hop_arr)
    {
        // ---- Validate input ----
        auto hop = hop_arr.unchecked<1>();
        int n = static_cast<int>(hop.shape(0));
        if (n != hop_) {
            throw std::runtime_error(
                "hop size mismatch: expected " + std::to_string(hop_) +
                ", got " + std::to_string(n));
        }

        // ---- Allocate output arrays (owned by numpy) ----
        auto mag_arr = py::array_t<float>(n_bins_);
        auto flux_arr = py::array_t<float>(n_bins_);
        float* magnitude = mag_arr.mutable_data();
        float* flux = flux_arr.mutable_data();

        // Zero-init magnitude (bins with no FFT data stay at 0)
        std::memset(magnitude, 0, n_bins_ * sizeof(float));

        // ---- Per-octave processing ----
        for (int oct = 0; oct < n_octaves_; ++oct) {
            auto& cfg = octaves_[oct];
            int fft_size = cfg.fft_size;
            float* buf = cfg.buffer.data();

            // 1. Slide buffer: shift left by n, append new samples
            //    This is a memmove + memcpy — the fastest way to do it.
            std::memmove(buf, buf + n, (fft_size - n) * sizeof(float));
            std::memcpy(buf + (fft_size - n), hop.data(0), n * sizeof(float));

            // 2. Apply window and compute rfft
            //    We use pre-allocated scratch buffers — no heap allocation here.
            //    The scratch is sized for the largest FFT, so it always fits.
            for (int i = 0; i < fft_size; ++i) {
                scratch_windowed_[i] = buf[i] * cfg.window[i];
            }

            // pocketfft r2c transform
            // Input:  real float[fft_size]
            // Output: complex float[fft_size/2 + 1]
            int n_rfft = fft_size / 2 + 1;

            pocketfft::shape_t shape = {static_cast<size_t>(fft_size)};
            pocketfft::stride_t stride_in = {sizeof(float)};
            pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};
            pocketfft::shape_t axes = {0};

            pocketfft::r2c(shape, stride_in, stride_out, axes,
                           pocketfft::FORWARD,
                           scratch_windowed_.data(), scratch_spectrum_.data(), 1.0f);

            // 3. Magnitude: |spectrum[k]| / fft_size
            //    Gather only the bins in this octave's mask.
            int n_masked = cfg.n_fft_bins;
            float inv_fft = 1.0f / fft_size;
            for (int i = 0; i < n_masked; ++i) {
                int k = cfg.mask_indices[i];
                scratch_masked_[i] = std::abs(scratch_spectrum_[k]) * inv_fft;
            }

            // 4. Rebin to bins_per_octave output bins
            int out_start = oct * bpo_;

            if (n_masked >= bpo_) {
                // More FFT bins than output bins: RMS-average groups
                // Same as Python: usable = (n_masked // bpo) * bpo
                int chunk_size = n_masked / bpo_;
                int usable = chunk_size * bpo_;

                for (int b = 0; b < bpo_; ++b) {
                    float sum_sq = 0.0f;
                    int base = b * chunk_size;
                    for (int j = 0; j < chunk_size; ++j) {
                        float v = scratch_masked_[base + j];
                        sum_sq += v * v;
                    }
                    magnitude[out_start + b] = std::sqrt(sum_sq / chunk_size);
                }
            } else if (n_masked > 0) {
                // Fewer FFT bins than output: spread directly
                for (int i = 0; i < n_masked; ++i) {
                    magnitude[out_start + i] = scratch_masked_[i];
                }
            }
        }

        // ---- Flux: half-wave rectified spectral difference ----
        if (has_prev_) {
            for (int i = 0; i < n_bins_; ++i) {
                float diff = magnitude[i] - prev_spectrum_[i];
                flux[i] = diff > 0.0f ? diff : 0.0f;
            }
        } else {
            std::memset(flux, 0, n_bins_ * sizeof(float));
            has_prev_ = true;
        }

        // Save current spectrum for next frame's flux
        std::memcpy(prev_spectrum_.data(), magnitude, n_bins_ * sizeof(float));

        // ---- Zero-crossing rate ----
        float zcr = 0.0f;
        int crossings = 0;
        for (int i = 1; i < n; ++i) {
            // signbit: true if negative
            if (std::signbit(hop(i)) != std::signbit(hop(i - 1))) {
                ++crossings;
            }
        }
        zcr = static_cast<float>(crossings) / n;

        return std::make_tuple(std::move(mag_arr), std::move(flux_arr), zcr);
    }

    void reset() {
        for (auto& cfg : octaves_) {
            std::fill(cfg.buffer.begin(), cfg.buffer.end(), 0.0f);
        }
        std::fill(prev_spectrum_.begin(), prev_spectrum_.end(), 0.0f);
        has_prev_ = false;
    }

    int n_bins() const { return n_bins_; }

private:
    int sr_, hop_, n_octaves_, bpo_;
    float fmin_;
    int n_bins_;
    std::vector<OctaveConfig> octaves_;
    std::vector<float> prev_spectrum_;
    bool has_prev_ = false;

    // Scratch buffers — pre-allocated, reused every push_hop call.
    // Sized for the largest FFT in the bank.
    std::vector<float> scratch_windowed_;
    std::vector<std::complex<float>> scratch_spectrum_;
    std::vector<float> scratch_masked_;
};


// ---------------------------------------------------------------------------
// Python module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_native, m) {
    m.doc() = "C++ native acceleration for flame-sheep-audio";

    py::class_<OctaveBankNative>(m, "OctaveBankNative",
        "C++ implementation of OctaveBankEngine — per-octave FFT bank.")

        .def(py::init<int, int, int, int, float>(),
             py::arg("sr") = 48000,
             py::arg("hop") = 512,
             py::arg("n_octaves") = 9,
             py::arg("bins_per_octave") = 12,
             py::arg("fmin") = 32.7f,
             "Construct the octave bank. Same parameters as the Python version.")

        .def("push_hop", &OctaveBankNative::push_hop,
             py::arg("hop"),
             "Process one hop of audio. Returns (magnitude, flux, zcr).")

        .def("reset", &OctaveBankNative::reset,
             "Reset all buffers and state.")

        .def_property_readonly("n_bins", &OctaveBankNative::n_bins,
             "Total number of output bins (n_octaves * bins_per_octave).");
}
