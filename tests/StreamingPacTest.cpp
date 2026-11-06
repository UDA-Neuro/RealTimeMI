#include "StreamingPac.h"

#include <iostream>
#include <limits>

namespace
{
constexpr double pi = 3.14159265358979323846;
constexpr int sampleRate = 1000;
constexpr int taps = 3001;
constexpr int window = 10000;

void Check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::optional<double> Feed(StreamingPac& pac, double depth, int samples)
{
    for (int n = 0; n < samples; ++n)
    {
        const double time = static_cast<double>(n) / sampleRate;
        const double phase = 2 * pi * 3.5 * time;
        const double carrier = std::cos(2 * pi * 150 * time);
        pac.Push(std::cos(phase) + (1 + depth * std::cos(phase)) * carrier);
    }
    return pac.Value();
}
}

int main()
{
    try
    {
        RollingModulationIndex index(4, 2);
        for (int n = 0; n < 2; ++n)
        {
            index.Push(-pi / 2, 0);
            index.Push(pi / 2, 2);
        }
        Check(index.Value().value_or(-1) == 1, "Concentration in one bin must give MI = 1");
        for (int n = 0; n < 2; ++n)
        {
            index.Push(-pi / 2, 1);
            index.Push(pi / 2, 1);
        }
        Check(std::abs(index.Value().value_or(-1)) < 1e-12, "Uniform replacement must give MI = 0");
        index.Push(pi / 2, 1);
        index.Push(pi / 2, 1);
        index.Push(pi / 2, 1);
        Check(!index.Value(), "Window with an empty bin must be invalid");

        StreamingPac pac(sampleRate, {3, 4}, {80, 250}, taps, window);
        Check(!Feed(pac, 0.8, taps + window - 2), "MI must be invalid before warm-up");
        pac.Reset();
        const auto coupled = Feed(pac, 0.8, taps + window - 1);
        Check(coupled && *coupled > 0.025, "Coupled signal should produce positive MI");
        pac.Push(std::numeric_limits<double>::quiet_NaN());
        Check(!pac.Value(), "Non-finite input must clear readiness");
        const auto uncoupled = Feed(pac, 0.0, taps + window - 1);
        Check(uncoupled && *uncoupled < 0.001, "Uncoupled signal should have near-zero MI");
        std::cout << "Coupled MI = " << *coupled << ", uncoupled MI = " << *uncoupled << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
