"""Small, explicit readers for the pinned EDF and BrainVision validation files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_edf(path: str | Path) -> tuple[np.ndarray, int, list[str]]:
    """Read equal-rate EEG channels; skip lower-rate EDF annotation signals."""
    path = Path(path)
    with path.open("rb") as stream:
        fixed = stream.read(256)
        def number(start, width):
            return float(fixed[start:start + width].decode("ascii").strip())
        header_bytes = int(number(184, 8))
        records = int(number(236, 8))
        record_seconds = number(244, 8)
        channels = int(number(252, 4))
        rest = stream.read(header_bytes - 256)
        if len(rest) != header_bytes - 256 or records < 1 or record_seconds <= 0:
            raise ValueError("Incomplete or invalid EDF header")
        cursor = 0
        def field(width):
            nonlocal cursor
            result = [rest[cursor + i * width:cursor + (i + 1) * width]
                      .decode("latin-1").strip() for i in range(channels)]
            cursor += channels * width
            return result
        labels = field(16)
        field(80)  # transducer
        field(8)   # physical dimension
        physical_min = np.array([float(x) for x in field(8)])
        physical_max = np.array([float(x) for x in field(8)])
        digital_min = np.array([float(x) for x in field(8)])
        digital_max = np.array([float(x) for x in field(8)])
        field(80)  # prefiltering
        samples = [int(x) for x in field(8)]
        selected = [i for i, label in enumerate(labels) if label != "EDF Annotations"]
        if not selected or len({samples[i] for i in selected}) != 1:
            raise ValueError("EDF EEG channels have differing sample rates")
        fs = samples[selected[0]] / record_seconds
        if not fs.is_integer():
            raise ValueError("EDF sample rate is not an integer")
        count = records * sum(samples)
        raw = np.fromfile(stream, dtype="<i2", count=count)
    if raw.size != count or np.any(digital_max[selected] == digital_min[selected]):
        raise ValueError("Incomplete EDF data or invalid digital scaling")
    raw = raw.reshape(records, sum(samples))
    offsets = np.cumsum([0] + samples)
    eeg = np.stack([raw[:, offsets[i]:offsets[i+1]] for i in selected], axis=-1)
    eeg = eeg.reshape(-1, len(selected))
    scale = (physical_max[selected] - physical_min[selected]) / (digital_max[selected] - digital_min[selected])
    values = (eeg - digital_min[selected]) * scale + physical_min[selected]
    return values.astype(np.float32), int(fs), [labels[i] for i in selected]


def read_brainvision(path: str | Path) -> tuple[np.ndarray, int, list[str]]:
    """Read the multiplexed float32 BrainVision format used by on003498."""
    path = Path(path)
    header = path.read_text(encoding="utf-8-sig")
    def setting(key):
        match = re.search(rf"^{re.escape(key)}=(.+)$", header, re.M)
        if not match:
            raise ValueError(f"Missing BrainVision {key}")
        return match.group(1).strip()
    if setting("DataOrientation") != "MULTIPLEXED" or setting("BinaryFormat") != "IEEE_FLOAT_32":
        raise ValueError("Unsupported BrainVision binary layout")
    channels = int(setting("NumberOfChannels"))
    fs = 1_000_000 / float(setting("SamplingInterval"))
    if not fs.is_integer():
        raise ValueError("BrainVision sample rate is not an integer")
    labels = []
    for index in range(1, channels + 1):
        labels.append(setting(f"Ch{index}").split(",", 1)[0].replace(r"\1", ","))
    binary = path.with_name(setting("DataFile"))
    if binary.stat().st_size % (4 * channels):
        raise ValueError("Incomplete BrainVision data")
    data = np.memmap(binary, dtype="<f4", mode="r").reshape(-1, channels)
    return data, int(fs), labels
