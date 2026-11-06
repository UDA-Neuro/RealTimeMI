#include "ModulationIndexFilter.h"

#include "BCIError.h"

RegisterFilter(ModulationIndexFilter, 2.A);

ModulationIndexFilter::ModulationIndexFilter()
{
    BEGIN_PARAMETER_DEFINITIONS
        "Filtering:ModulationIndex int MIChannel= 1 1 1 % // one-based input channel",
        "Filtering:ModulationIndex float MIPhaseLow= 3Hz 3Hz 0Hz % // phase band lower edge",
        "Filtering:ModulationIndex float MIPhaseHigh= 4Hz 4Hz 0Hz % // phase band upper edge",
        "Filtering:ModulationIndex float MIAmplitudeLow= 80Hz 80Hz 0Hz % // amplitude band lower edge",
        "Filtering:ModulationIndex float MIAmplitudeHigh= 250Hz 250Hz 0Hz % // amplitude band upper edge",
        "Filtering:ModulationIndex int MIFirTaps= 3001 3001 3 8193 // odd FIR length in samples",
        "Filtering:ModulationIndex int MIWindowSamples= 10000 10000 2 1000000 // rolling window length in samples",
        "Filtering:ModulationIndex int MIPhaseBins= 18 18 2 360 // number of phase bins",
    END_PARAMETER_DEFINITIONS
}

ModulationIndexFilter::Settings ModulationIndexFilter::ReadSettings() const
{
    return {
        static_cast<int>(Parameter("MIChannel")), Parameter("SamplingRate").InHertz(),
        {Parameter("MIPhaseLow").InHertz(), Parameter("MIPhaseHigh").InHertz()},
        {Parameter("MIAmplitudeLow").InHertz(), Parameter("MIAmplitudeHigh").InHertz()},
        static_cast<int>(Parameter("MIFirTaps")), static_cast<int>(Parameter("MIWindowSamples")),
        static_cast<int>(Parameter("MIPhaseBins"))
    };
}

void ModulationIndexFilter::Preflight(const SignalProperties& input, SignalProperties& output) const
{
    const auto s = ReadSettings();
    if (input.Elements() < 1)
        bcierr << "RealTimeMI requires at least one sample per input block";
    if (s.channel < 1 || s.channel > input.Channels())
        bcierr << "MIChannel must select an existing input channel";
    if (!std::isfinite(s.rate) || !(0 < s.phase.low && s.phase.low < s.phase.high
        && s.phase.high < s.amplitude.low && s.amplitude.low < s.amplitude.high
        && s.amplitude.high < s.rate / 2))
        bcierr << "MI bands must be positive, ordered, disjoint, and below Nyquist";
    if (s.taps < 3 || s.taps > 8193 || s.taps % 2 == 0)
        bcierr << "MIFirTaps must be odd and between 3 and 8193";
    if (s.bins < 2 || s.bins > 360 || s.window < s.bins || s.window > 1000000)
        bcierr << "MIPhaseBins must be between 2 and 360 and MIWindowSamples must be between MIPhaseBins and 1000000";

    output = SignalProperties(2, 1, SignalType::float32);
    output.ChannelLabels()[0] = "MI";
    output.ChannelLabels()[1] = "MIValid";
    output.SetUpdateRate(input.UpdateRate());
}

void ModulationIndexFilter::Initialize(const SignalProperties&, const SignalProperties&)
{
    const auto s = ReadSettings();
    mChannel = s.channel - 1;
    mPac.emplace(s.rate, s.phase, s.amplitude, s.taps, s.window, s.bins);
}

void ModulationIndexFilter::StartRun()
{
    mPac->Reset();
}

void ModulationIndexFilter::Process(const GenericSignal& input, GenericSignal& output)
{
    for (int sample = 0; sample < input.Elements(); ++sample)
        mPac->Push(input(mChannel, sample));

    const auto value = mPac->Value();
    output(0, 0) = value.value_or(0.0);
    output(1, 0) = value.has_value() ? 1 : 0;
}
