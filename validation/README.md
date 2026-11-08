# Public-data validation pilot (2026-11-08)

This folder contains reproducible **exploratory** checks for RealTimeSpike,
RealTimeHFO, and RealTimeMI. Raw recordings, converted NPZ files, event CSVs,
and manifests are stored under ignored `build/validation/`; no EEG bytes are
committed. The two detector packages must be installed or checked out under
`build/RealTimeSpike` and `build/RealTimeHFO` as in this workspace.

## Pinned sources and acquisition

- Spike: NEMAR `nm000360` v1.0.2, subjects 01–03, EDF at 1 kHz. The 25-subject
  dataset is [described here](https://nemar.org/dataset/nm000360); its license
  is CC BY-NC 4.0. Its event labels mark time and side, not a unique contact.
- HFO: NEMAR/OpenNeuro `on003498` v1.0.0, subject 01, run 01. BrainVision
  data are 2 kHz, 50 monopolar channels. The [dataset page](https://www.nemar.org/dataset/on003498)
  lists ripple, fast-ripple, and combined automated event marks. The 120 MB
  `.eeg` file was obtained from the original OpenNeuro S3 object because the
  NEMAR mirror returned HTTP 500; its MD5 matched the pinned manifest
  (`b19be9e7ba854fc2b2783a9863052b7c`). This is an agreement test against
  published automated marks, **not** a test against an independent expert
  ground truth.

To acquire the same files after checking out this repository, run:

```powershell
python validation/download_public_subset.py spike --subject 01 --signal
python validation/download_public_subset.py spike --subject 02 --signal
python validation/download_public_subset.py spike --subject 03 --signal
python validation/download_public_subset.py hfo --subject 01 --run 01 --signal
```

## Locked pilot matching rules

- Spike: run the released detector on every 5-second span of each full
  181-second recording. Cluster same-side channel detections within 50 ms.
  One-to-one match a prediction to an annotation on the same side if peak
  times differ by at most 100 ms. This assesses time/side agreement, not
  electrode-level accuracy. Unannotated candidates are counted as unmatched,
  though the annotations may not include every physiologic event.
- HFO: reconstruct each annotated bipolar pair from its two monopolar
  channels, anti-alias downsample to 1 kHz, and run the released detector on
  the full 300-second recording. Score only pure `ripple` marks on the exact
  pair with interval overlap plus 10 ms. Exclude the first/last 0.5 seconds.
  Unmatched detections inside `fr` or `frandr` intervals are ignored; these
  classes are outside this ripple-only comparison.
- MI: inject graded 3.5/130 Hz coupling into fixed-seed 1 kHz synthetic data
  and compare the GPU Tort-type normalized KL result with an independently
  coded SciPy convolution using the same analytic FIR and bins. Repeat the
  numerical comparison on four channels in the first 30 seconds of public
  `nm000360` subject 01. This does **not** yet compare against winPACT or prove
  physiologic validity.

```powershell
python validation/evaluate_spike.py 01 02 03
python validation/evaluate_hfo.py
python validation/evaluate_mi.py
python -m unittest discover -s validation -p 'test_*.py'
```

## Pilot observations

| Toolbox | Public-data result | Interpretation |
| --- | --- | --- |
| Spike | 104/108 annotated time/side events matched across three subjects (sensitivity 0.963); 104/412 clustered predictions matched (precision 0.252). | High recall with many unmatched candidates. These three subjects are exploratory and cannot establish clinical precision. |
| HFO | 109/3,628 pure-ripple marks matched (recall 0.030); 109/113 scored candidates matched (precision 0.965). | The current threshold is far less sensitive than the published marking method on this run. No equivalence claim is supported. |
| MI | Synthetic MI rose monotonically with injected coupling (30-second means about 0.000003, 0.00538, 0.02165, 0.05082). Maximum GPU-versus-CPU difference on four public channels was 1.14×10⁻⁷. | Numerical agreement for the tested settings; separate algorithm and artifact controls remain. |

The MI public-data check exposed severe slowdown for Fortran-strided input
after channel selection. RealTimeMI now converts input to contiguous sample
major layout, and a GPU regression test covers this case.

## Next validation stage

Treat these subjects as development/exploration data. Freeze any detector
changes before evaluating held-out patients. Compare to the exact reference
algorithms with identical montage, bands, and event-matching rules; obtain
expert review of a random sample of matched and unmatched detections. Report
per-patient estimates and uncertainty rather than only pooled event counts.
