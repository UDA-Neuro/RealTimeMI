"""One-window, label-ordered color map of recording-wide MI z-scores."""

from __future__ import annotations

import csv
import math
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk


def _color(z: float) -> str:
    if not math.isfinite(z):
        return "#b8b8b8"
    t = min(abs(z) / 3, 1)
    neutral = (250, 250, 250)
    end = (190, 44, 53) if z >= 0 else (40, 90, 190)
    return "#" + "".join(f"{round(a * (1 - t) + b * t):02x}"
                         for a, b in zip(neutral, end))


def show_map(parent: tk.Misc, summary_path: str | Path) -> tk.Toplevel:
    with Path(summary_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    shown = [r for r in rows if r["electrode_in_map"] == "1"]
    excluded = [r["label"] for r in rows if r["electrode_in_map"] != "1"]
    window = tk.Toplevel(parent)
    window.title("RealTimeMI — Mean MI z-score map")
    window.geometry("1050x850")
    frame = ttk.Frame(window, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Mean MI across the analyzed recording → z-score across electrodes",
              font=("Segoe UI", 12, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Cells follow channel labels; positions do not represent physical electrode locations.").pack(anchor="w", pady=(4, 0))
    if excluded:
        ttk.Label(frame, text="Excluded from z-score: " + ", ".join(excluded)).pack(anchor="w")
    canvas = tk.Canvas(frame, background="white", highlightthickness=0)
    canvas.pack(fill="both", expand=True, pady=(10, 0))
    canvas.create_text(15, 12, anchor="nw", text="Blue: below electrode mean     White: mean     Red: above electrode mean     Gray: undefined")
    groups: dict[str, list[dict]] = {}
    for row in shown:
        match = re.fullmatch(r"([A-Za-z]+)(\d+)", row["label"])
        group = match.group(1).upper() if match and match.group(1).upper() in ("A", "B") else "Channels"
        groups.setdefault(group, []).append(row)
    y = 46
    cell_w, cell_h, gap = 94, 39, 5
    for group, entries in groups.items():
        columns = 8 if group in ("A", "B") else 10
        canvas.create_text(15, y, anchor="nw", text=f"{group} ({len(entries)})", font=("Segoe UI", 11, "bold"))
        y += 30
        for index, row in enumerate(entries):
            col, line = index % columns, index // columns
            x0 = 15 + col * (cell_w + gap)
            y0 = y + line * (cell_h + gap)
            z = float(row["z_score"])
            canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h,
                                    fill=_color(z), outline="#999")
            label = row["label"]
            value = f"{z:+.2f}" if math.isfinite(z) else "N/A"
            canvas.create_text(x0 + cell_w / 2, y0 + cell_h / 2,
                               text=f"{label}  {value}", font=("Segoe UI", 9))
        y += math.ceil(len(entries) / columns) * (cell_h + gap) + 18
    canvas.configure(scrollregion=(0, 0, 1020, y))
    if y > 750:
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)
    ttk.Label(frame, text=f"Summary CSV: {summary_path}", wraplength=1000).pack(anchor="w")
    return window
