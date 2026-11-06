#pragma once

#include "GenericFilter.h"
#include "StreamingPac.h"

class ModulationIndexFilter : public GenericFilter
{
public:
    ModulationIndexFilter();
    void Preflight(const SignalProperties&, SignalProperties&) const override;
    void Initialize(const SignalProperties&, const SignalProperties&) override;
    void StartRun() override;
    void Process(const GenericSignal&, GenericSignal&) override;

private:
    struct Settings
    {
        int channel;
        double rate;
        StreamingPac::Band phase, amplitude;
        int taps, window, bins;
    };

    Settings ReadSettings() const;
    int mChannel = 0;
    std::optional<StreamingPac> mPac;
};
