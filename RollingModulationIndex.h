#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

// Consumes time-aligned phase (radians) and amplitude samples.
class RollingModulationIndex
{
public:
    explicit RollingModulationIndex(std::size_t windowSamples, std::size_t phaseBins = 18)
        : mWindow(windowSamples), mBins(phaseBins)
    {
        if (phaseBins < 2 || windowSamples < phaseBins)
            throw std::invalid_argument("Window must have at least one sample per phase bin");
    }

    void Reset()
    {
        std::fill(mBins.begin(), mBins.end(), Bin{});
        mNext = 0;
        mSize = 0;
    }

    void Push(double phase, double amplitude)
    {
        if (!std::isfinite(phase) || !std::isfinite(amplitude) || amplitude < 0)
        {
            Reset();
            return;
        }

        constexpr double pi = 3.14159265358979323846;
        double wrapped = std::remainder(phase, 2 * pi);
        if (wrapped == pi)
            wrapped = -pi;
        const double fraction = (wrapped + pi) / (2 * pi);
        const auto index = std::min(static_cast<std::size_t>(fraction * mBins.size()), mBins.size() - 1);

        auto& slot = mWindow[mNext];
        if (mSize == mWindow.size())
        {
            auto& old = mBins[slot.bin];
            if (--old.count == 0)
                old.sum = 0;
            else
                old.sum -= slot.amplitude;
        }
        else
            ++mSize;

        slot = {index, amplitude};
        mNext = (mNext + 1) % mWindow.size();
        auto& bin = mBins[index];
        ++bin.count;
        bin.sum += amplitude;
    }

    std::optional<double> Value() const
    {
        if (mSize < mWindow.size())
            return std::nullopt;

        long double total = 0;
        for (const auto& bin : mBins)
        {
            if (bin.count == 0)
                return std::nullopt;
            total += std::max(0.0L, bin.sum / bin.count);
        }
        if (!(total > 0) || !std::isfinite(total))
            return std::nullopt;

        const auto logBinCount = std::log(static_cast<long double>(mBins.size()));
        long double entropy = 0;
        for (const auto& bin : mBins)
        {
            const auto probability = (bin.sum / bin.count) / total;
            if (probability > 0)
                entropy -= probability * std::log(probability);
        }
        return static_cast<double>(std::clamp(1 - entropy / logBinCount, 0.0L, 1.0L));
    }

private:
    struct Sample
    {
        std::size_t bin = 0;
        double amplitude = 0;
    };

    struct Bin
    {
        std::size_t count = 0;
        long double sum = 0;
    };

    std::vector<Sample> mWindow;
    std::vector<Bin> mBins;
    std::size_t mNext = 0;
    std::size_t mSize = 0;
};
