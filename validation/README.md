# Public-data validation pilot (2026-11-08)

This folder contains reproducible **exploratory** checks for RealTimeSpike,
RealTimeHFO, and RealTimeMI. Raw recordings, converted NPZ files, event CSVs,
and manifests are stored under ignored `build/validation/`; no EEG bytes are
committed. The two detector packages must be installed or checked out under
`build/RealTimeSpike` and `build/RealTimeHFO` as in this workspace.

## Pinned sources and acquisition

- Spike: NEMAR `nm000360` v1.0.2, subjects 01–03, EDF at 1 kHz. The 25-subject
  dataset is [described here](https://nemar.org/dataset/nm000360); its license
  is CC BY-NC 4.0. Its main event labels mark time and side. The separate
  `derivatives/*events_interpretation.tsv` files list the contacts involved in
  each event; use these for contact-level assessment.
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
  times differ by at most 100 ms. This initial score assesses time/side
  agreement. The diagnosis below separately checks the published contact
  lists. Unannotated candidates are counted as unmatched, though the
  annotations may not include every physiologic event.
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
python validation/diagnose_disagreement.py
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

## Disagreement audit (same pilot data)

The reproducible audit is `diagnose_disagreement.py`; detailed output is
written to ignored `build/validation/diagnosis.json`. It does not alter the
released detector or tune settings on this benchmark.

- HFO: of 3,628 published pure-ripple marks, only 1,050 overlap any RMS
  crossing above the released mean + 5 SD threshold (with 10 ms margin).
  The threshold excludes 2,578 (71.1%). Of the 1,050, 795 overlap a merged
  candidate with an allowed 6–200 ms duration; only 109 also meet the
  four-positive-peak condition *within the above-threshold span*. The median
  maximum RMS within a marked event is 0.791 times the local threshold.
  Running the same detector directly at the original 2 kHz gives 107/3,628
  matches versus 109/3,628 after anti-alias downsampling to 1 kHz. Thus the
  downsampling is not the main explanation on this recording. The original
  publication describes a different detector that selects quiet baseline via
  time-frequency entropy, then thresholds the filtered envelope for at least
  20 ms; it does not use this detector's mean + 5 SD / four-peak rule.
- Spike: the main time/side score matches 104/108 annotations, but 308/412
  clustered candidates remain unmatched. Across subjects, matched clusters
  have median peak/threshold ratios of 3.50, 4.13, and 2.96; unmatched
  clusters have 1.18, 1.22, and 1.14. The corresponding median numbers of
  channels per cluster are 5, 11, 4 versus 1.5, 2, 1. Subject 02 has 38
  left-sided unmatched clusters but no left-sided expert marks. This suggests
  the simple 10–60 Hz envelope detector also admits weak, spatially sparse
  transients; it does not prove they are artifacts or false IEDs. The expert
  reference paper used a different feature-based model and specifies contact
  lists in derivatives. Checking those lists finds a prediction on at least
  one listed contact for 100/108 annotations, and 306/327 listed
  event-contact pairs matched within 100 ms. These are exploratory checks on
  three subjects and are not independent clinical performance estimates.

The source datasets and algorithm descriptions are [HFO data](https://www.nemar.org/dataset/on003498),
[original HFO study](https://www.nature.com/articles/s41598-017-13064-1.pdf),
and the [IED dataset paper](https://www.nature.com/articles/s41597-024-04187-y.pdf).

## Annotation-free interictal detector comparison

The next comparison uses [Kuroda et al. OpenNeuro ds006107 v1.0.0](https://openneuro.org/datasets/ds006107/versions/1.0.0),
mirrored at [NEMAR](https://www.nemar.org/dataset/on006107). This is 1 kHz
slow-wave-sleep iEEG from epilepsy patients, used for interictal high-frequency
analysis. The pinned manifest has no event annotation files. Subjects 01, 04,
and 06 were selected before event analysis. The first 300 s and eight evenly
spaced EDF channels per subject were analyzed, with no rereferencing. EDF
SHA-256 digests are checked against the manifest. Raw data and output event
CSVs remain under ignored `build/validation/`.

The independent comparator is [epycom 0.3](https://pypi.org/project/epycom/):
its RMS HFO detector and Janca spike detector. The benchmark script loads
only those two unmodified detection functions because epycom's umbrella import
requires an unrelated optional numba module. Both HFO implementations receive
the same 80–250 Hz, fourth-order zero-phase Butterworth filtered signal. They
use a 5 ms *effective* RMS window, mean + 5 SD threshold, gaps merged through
10 ms, 6–200 ms duration, and at least four positive peaks. epycom's RMS
helper returns the first value of a centered convolution: its nominal
`window_size=5` uses three actual signal samples, while `window_size=9` uses
five. The script reports both the nominal-5 and effective-5 comparisons. It
also converts epycom's window-end timestamp to the same event boundary
convention before applying the shared duration/peak rules. epycom's native
RMS function only returns candidate spans; the common merge, duration, and
peak rules are supplied by the comparison adapter. The effective-5 result
tests RMS candidate parity under identical postprocessing, not agreement
between two independently complete HFO classifiers.

For spikes, both see the same EDF channel, 10–60 Hz band, and no notch. epycom
0.3 hard-codes a 50 Hz notch despite exposing `line_freq`; the benchmark
disables that optional notch in the comparator so the shared input remains
unnotched. The remaining threshold definitions **cannot be set equal**:
RealTimeSpike uses median + 6 robust SD of the Hilbert envelope, while Janca
fits a log-normal envelope baseline with `k1=k2=3.65` by default. The spike
comparison therefore measures agreement at stated, different thresholds.
All detectors run once on each full 300 s segment; this checks algorithmic
agreement, not the released application's 5 s incremental processing latency.

| Subject, 300 s × 8 channels | HFO current / epycom effective-5 / matched | HFO nominal-5 matched | Spike current / Janca / matched |
| --- | ---: | ---: | ---: |
| 01 | 48 / 48 / 48 | 37 | 210 / 109 / 82 |
| 04 | 11 / 11 / 11 | 9 | 185 / 99 / 96 |
| 06 | 4 / 4 / 4 | 3 | 147 / 62 / 56 |
| **Total** | **63 / 63 / 63** | **49** | **542 / 270 / 234** |

Events are matched one-to-one on the same channel: HFO intervals may overlap
within a 10 ms margin; spike peaks may differ by at most 100 ms. The full
[machine-readable pilot](unannotated_pilot.json) records the source hashes,
selected channels, package versions, and per-subject event Jaccard scores.
No sensitivity, specificity, or false-positive rate can be inferred without
independent event annotations. Eight channels per patient make this an initial
implementation check, not an all-electrode or multi-center validation.

To reproduce from a checkout with the RealTimeHFO and RealTimeSpike source
trees under `build/`, run:

```powershell
python validation/download_public_subset.py kuroda --subject 01 --signal
python validation/download_public_subset.py kuroda --subject 04 --signal
python validation/download_public_subset.py kuroda --subject 06 --signal
python -m pip install --no-deps --target build/validation/comparators/epycom_pkg epycom==0.3
python validation/compare_unannotated.py 01 04 06 --seconds 300 --channels 8
```

The pilot ran with Python 3.12, NumPy 2.3.5, and SciPy 1.18.1. epycom's
published dependency metadata requests NumPy <1.27; only the two audited
detector modules were loaded and executed with the stated local versions.

## Next validation stage

Treat these subjects as development/exploration data. Freeze any detector
changes before evaluating held-out patients. Compare to the exact reference
algorithms with identical montage, bands, and event-matching rules; obtain
expert review of a random sample of matched and unmatched detections. Report
per-patient estimates and uncertainty rather than only pooled event counts.

