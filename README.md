# Real-Time Modulation Index Toolbox

RealTimeMI is a local, Python-led modulation index toolbox for completed BESA
ECoG recordings. It uses a bundled native BESA decoder, SciPy anti-alias
resampling, and a local NVIDIA GPU via CuPy. MATLAB and BCI2000 are not needed
to run the standalone path. A separate BCI2000 module and MATLAB reference
implementation remain in this repository. The estimator follows
[Tort et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/20463205/).

## Standalone quick start

On Windows with a supported NVIDIA GPU and driver, install Python 3.12 and:

```powershell
python -m pip install ".[gpu]"
python -m realtimemi
```

The GUI lets you choose a complete `.besa` file or the supplied `.npz` demo
and an output folder, runs
the default first five-minute segment, plots MI for a selected channel, and
writes timestamped `mi-*.csv` and `mi-*.json` files. For command-line use:

```powershell
python -m realtimemi --input 'D:\data\recording.besa' --output-dir 'D:\results'
```

To try the public synthetic demo with no patient data, run:

```powershell
python -m realtimemi --input demo/synthetic_100ch_5min_1khz.npz --output-dir mi-output
```

The demo contains 100 channels × 300 seconds sampled at 1 kHz. It has
deliberate 3.5 Hz to 120 Hz coupling that varies by channel. The waveform
repeats a two-second template to make the download small. It is fictional
test data, not physiological validation. Recreate it with
`python demo/generate_demo.py`. NPZ input skips BESA decoding and resampling;
the 30-second timing target must be assessed separately on real BESA input.

The bundled decoder DLL currently targets Windows x64. This release path has
been exercised on a 10 kHz compressed BESA recording; other rates and BESA
variants need separate testing. The software is for research analysis. Inspect
channel quality and validate against an independent export before using its
results for consequential clinical decisions. Never add patient recordings or
generated analysis files to the public repository.

## First release target

Process each five-minute recording after it ends and return results within
30 seconds, including file reading, resampling, MI calculation, and result
saving. The initial input target is the research tower's BESA recording. The
GPU analysis uses 30-second windows, 50% overlap, and the 3–4 Hz phase and
80–250 Hz amplitude bands. A five-minute recording yields 19 values per
channel, with windows starting every 15 seconds.

For 10 kHz input, use a documented anti-alias low-pass filter before reducing
the sample rate to 1 kHz. Do not discard every tenth sample without filtering.
Keep the resampling filter, edge handling, channel selection, MI algorithm,
phase-bin count, and software versions fixed in the result metadata. The first
performance gate is a full end-to-end run on a real five-minute BESA segment,
repeated at least three times, with each run under 30 seconds. Verify the
decoded signal and MI against an independent export/reference before treating
the result as scientifically validated.

To repeat the standalone timing test on a local recording:

```powershell
python offline/benchmark_python_standalone.py --input 'PATH_TO_RECORDING.besa' --repeats 3
```

The script writes `build/python-benchmark/benchmark.json` and the individual
MI results locally. Add `--reference PATH_TO_REFERENCE_MI.csv` to compare
finite values and the maximum absolute difference with a headerless MATLAB
reference CSV. No recording is uploaded by this command.

## Python standalone benchmark

On one local 10 kHz BESA recording (132 channels, first five minutes) using an
NVIDIA RTX A500 Laptop GPU, three complete Python runs took 15.721, 14.817,
and 23.246 seconds from decoding through CSV and JSON output. A separate
new-process CLI run took 16.632 seconds wall clock. Each run produced
19 windows and 2,506 finite MI values; the other two values were undefined.
All three Python result matrices matched exactly, including NaN positions.

The MATLAB reference produced the same finite-value mask. Among finite MI
values, the median absolute difference was 3.95e-17, the 95th percentile was
2.27e-4, and the maximum was 0.00386. The reason for the larger differences
in selected channels has not been established. These are performance and
cross-implementation checks on one recording, not validation against an
independent BESA waveform export. The benchmark starts from a finalized file;
file finalization and GUI rendering are outside the timed interval.

## BESA pilot benchmark

On the local 10 kHz, 132-channel research recording, the first five minutes
were decoded, anti-alias filtered, reduced to 1 kHz, processed on the GPU, and
saved to CSV. With eight parallel decoding workers and a native packed-data
decoder, the three end-to-end times were 15.712, 19.355, and 21.923 seconds.
Each run saved 19 windows by 132 channels; 2 of 2508 MI values were undefined.
The benchmark log is `build/besa-pipeline-benchmark.json` (ignored by Git).

The MATLAB GPU service was started before timing, as it must be while recording
for the 30-second post-recording target. Cold MATLAB startup took 75 seconds
in this run and is not included in the three times. The benchmark starts from
an already finalized BESA file. File finalization and a GUI display are not
included. This is a pilot result on one recording and machine.

The BESA packed-data decoder follows the public BESA format specification.
Native and Python decoding matched exactly across all channels in the first
five-second data block. Blockwise anti-alias resampling matched a whole-segment
reference exactly on the first ten seconds. The final native MI CSV matched
the Python-decoded MI CSV exactly. An independent SciPy Tort calculation on
three selected channels agreed within 1e-6; saturated channels showed larger
differences and need quality-control flags. An independent BESA application
export has not yet been compared against the decoded waveform.

To reproduce the performance test on a compatible Windows machine, build
`offline/besa_packed.c` as `build/besa_packed.dll` with Zig or another C
compiler, then run:

```powershell
python offline/benchmark_besa_pipeline.py --input 'PATH_TO_RECORDING.besa' --repeats 3 --workers 8 --fused --service --native
```

The native decoder, MATLAB Parallel Computing Toolbox, Signal Processing
Toolbox, and supported NVIDIA GPU are required for this benchmark path.

The standalone MATLAB prototype accepts a samples-by-channels matrix and
returns a windows-by-channels MI matrix and the window start and end times:

```matlab
addpath('offline')
[mi, windowSeconds] = recording_mi_gpu(data, sampleRate);
```

It requires a supported NVIDIA GPU, Parallel Computing Toolbox, and Signal
Processing Toolbox. The centered FIR uses reflection at the recording edges
to cover the first and last windows. Windows without a defined MI return
`NaN`. The benchmark can read a finalized BESA file automatically. Live file
detection and a quality-control display are under development.

## Build in BCI2000

Use a BCI2000 source checkout configured for a C++17-capable compiler. The
workspace copy under `.bci2000/BCI2000src` includes a custom CMake file that
adds this module. For another checkout, add the following line to
`src/custom/CMakeLists.txt`:

```cmake
add_subdirectory("D:/2026/REAL-TIME-MI-TOOLBOX/bci2000/RealTimeMI" RealTimeMI)
```

Regenerate the BCI2000 build, build the `RealTimeMI` target, and select its
executable as the Signal Processing module in the BCI2000 launcher. Use a source
that supplies continuous samples and an application that accepts a two-channel
control signal. See the [BCI2000 custom module guide](https://www.bci2000.org/mediawiki/index.php/Programming_Howto:Create_a_custom_Signal_Processing_Module).

## Parameters and output

| Parameter | Default |
| --- | --- |
| `MIChannel` | Input channel 1 |
| `MIPhaseLow`, `MIPhaseHigh` | 3–4 Hz |
| `MIAmplitudeLow`, `MIAmplitudeHigh` | 80–250 Hz |
| `MIFirTaps` | 3001 samples |
| `MIWindowSamples` | 10000 samples |
| `MIPhaseBins` | 18 |

The defaults are starting values for a 1000 Hz source. The 250 Hz upper band
edge must be below Nyquist, so the input rate must exceed 500 Hz with these
bands. Set the bands for the signal of interest and allow the amplitude band to include the modulation
sidebands. Choose the FIR length and window in samples for the actual source
rate. The two causal FIRs share a delay of `(MIFirTaps - 1) / 2` samples.
At 1000 Hz, that is 1.5 seconds, and the rolling window spans 10 seconds.
The first possible valid output follows
`MIFirTaps + MIWindowSamples - 1` input samples, about 13 seconds at the default
settings, rounded up to the next block boundary.

The float output is two channels by one element. `MI` is in [0, 1] and
`MIValid` is 1 when the window is full, all phase bins contain samples, and
the normalized amplitude distribution is defined. During warm-up or after
invalid input, an unavailable value is represented by `MI=0, MIValid=0`.

## Standalone verification

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The standalone check uses synthetic signals and needs no BCI2000 installation.
BCI2000 integration and processing time still need to be verified on the target
system.
