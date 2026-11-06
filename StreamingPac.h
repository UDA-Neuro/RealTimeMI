#pragma once

#include "RollingModulationIndex.h"

#include <complex>

class StreamingPac
{
public:
    struct Band
    {
        double low;
        double high;
    };

    StreamingPac(double sampleRate, Band phase, Band amplitude, std::size_t taps,
                 std::size_t windowSamples, std::size_t phaseBins = 18)
        : mPhaseFilter(MakeFilter(sampleRate, phase, taps)),
          mAmplitudeFilter(MakeFilter(sampleRate, amplitude, taps)),
          mHistory(taps), mIndex(windowSamples, phaseBins)
    {
        if (phase.high >= amplitude.low)
            throw std::invalid_argument("Phase band must be below the amplitude band");
    }

    void Reset()
    {
        mNext = 0;
        mFilled = 0;
        mIndex.Reset();
    }

    void Push(double sample)
    {
        if (!std::isfinite(sample))
        {
            Reset();
            return;
        }

        mHistory[mNext] = sample;
        if (++mNext == mHistory.size())
            mNext = 0;
        if (mFilled < mHistory.size() && ++mFilled < mHistory.size())
            return;

        std::complex<double> phase, amplitude;
        auto position = mNext;
        for (std::size_t tap = 0; tap < mHistory.size(); ++tap)
        {
            if (position == 0)
                position = mHistory.size();
            const auto input = mHistory[--position];
            phase += input * mPhaseFilter[tap];
            amplitude += input * mAmplitudeFilter[tap];
        }

        if (phase == std::complex<double>{})
            mIndex.Reset();
        else
            mIndex.Push(std::arg(phase), std::abs(amplitude));
    }

    std::optional<double> Value() const { return mIndex.Value(); }

private:
    static std::vector<std::complex<double>> MakeFilter(double rate, Band band, std::size_t taps)
    {
        if (!std::isfinite(rate) || !(0 < band.low && band.low < band.high && band.high < rate / 2)
            || taps < 3 || taps % 2 == 0)
            throw std::invalid_argument("Expected a band below Nyquist and an odd FIR length >= 3");

        constexpr double pi = 3.14159265358979323846;
        const double halfBandwidth = (band.high - band.low) / (2 * rate);
        const double center = (band.low + band.high) / (2 * rate);
        const double delay = static_cast<double>(taps / 2);
        std::vector<std::complex<double>> coefficients(taps);
        for (std::size_t tap = 0; tap < taps; ++tap)
        {
            const double time = static_cast<double>(tap) - delay;
            const double window = 0.5 + 0.5 * std::cos(pi * time / delay);
            const double lowpass = time == 0 ? 2 * halfBandwidth
                : std::sin(2 * pi * halfBandwidth * time) / (pi * time);
            const double angle = 2 * pi * center * time;
            coefficients[tap] = 2 * lowpass * window
                * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        return coefficients;
    }

    std::vector<std::complex<double>> mPhaseFilter, mAmplitudeFilter;
    std::vector<double> mHistory;
    RollingModulationIndex mIndex;
    std::size_t mNext = 0;
    std::size_t mFilled = 0;
};
