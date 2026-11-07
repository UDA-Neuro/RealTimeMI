"""Desktop waveform viewer for a growing BESA recording or file replay."""

from __future__ import annotations

import queue
import struct
import threading
import time
import tkinter as tk
import zlib
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from realtimemi.viewer import BesaBlockSource, WaveformChunk, append_waveform


class EEGViewer:
    def __init__(self, root: tk.Tk | tk.Toplevel):
        self.root = root
        root.title("RealTimeMI — BESA EEG Viewer")
        root.geometry("1100x700")
        self.path = tk.StringVar()
        self.mode = tk.StringVar(value="follow")
        self.channel = tk.StringVar(value="1")
        self.status = tk.StringVar(value="Choose a BESA file, then start the viewer.")
        self.events: queue.Queue = queue.Queue()
        self.running = False
        self.generation = 0
        self.first_channel = 0
        self.latest: WaveformChunk | None = None
        self.last_received = 0.0

        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="BESA recording").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.path).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse…", command=self.pick_file).grid(row=1, column=1)
        controls = ttk.Frame(frame)
        controls.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        ttk.Radiobutton(controls, text="Follow growing file", variable=self.mode,
                        value="follow").pack(side="left")
        ttk.Radiobutton(controls, text="Replay saved file (1×)", variable=self.mode,
                        value="replay").pack(side="left", padx=(20, 0))
        self.start_button = ttk.Button(controls, text="Start", command=self.start)
        self.start_button.pack(side="left", padx=(24, 4))
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(controls, text="First channel").pack(side="left", padx=(24, 5))
        self.channel_box = ttk.Combobox(controls, textvariable=self.channel, state="readonly", width=16)
        self.channel_box.pack(side="left")
        self.channel_box.bind("<<ComboboxSelected>>", self.change_channel)
        ttk.Label(frame, textvariable=self.status, wraplength=1000).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.canvas = tk.Canvas(frame, background="white", highlightthickness=1,
                                highlightbackground="#c5cbd0")
        self.canvas.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _: self.draw())
        ttk.Label(frame, text="Waveform display: 1 kHz antialias resampling; each channel autoscaled in ADC counts.").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(7, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=1)
        root.after(100, self.poll_events)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def pick_file(self):
        path = filedialog.askopenfilename(filetypes=[("BESA recordings", "*.besa"), ("All files", "*")])
        if path:
            self.path.set(path)

    def start(self):
        path = Path(self.path.get())
        if not path.is_file() or path.suffix.lower() != ".besa":
            messagebox.showerror("Input error", "Choose an existing .besa file")
            return
        self.stop()
        self.generation += 1
        generation = self.generation
        self.running = True
        self.latest = None
        self.draw()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        mode = self.mode.get()
        self.status.set("Waiting for a complete BESA data block…")

        def worker():
            source = BesaBlockSource(path, replay=mode == "replay")
            announced = False
            last_error = None
            while self.running and generation == self.generation:
                try:
                    chunk = source.poll(self.first_channel)
                    if source.labels and not announced:
                        self.events.put((generation, "labels", source.labels))
                        announced = True
                    if chunk is not None:
                        self.events.put((generation, "chunk", chunk))
                    elif mode == "replay" and source.next_index is not None and source.next_index >= 0:
                        # A replay reaches its end when no next block exists.
                        with path.open("rb") as stream:
                            from offline.besa_reader import metadata
                            _, _, _, blocks = metadata(stream)
                        if source.next_index >= len(blocks):
                            self.events.put((generation, "finished", None))
                            break
                    last_error = None
                except (OSError, ValueError, KeyError, UnicodeError,
                        struct.error, zlib.error, IndexError, StopIteration) as exc:
                    error = f"Waiting for a readable BESA block: {type(exc).__name__}: {exc}"
                    if error != last_error:
                        self.events.put((generation, "status", error))
                        last_error = error
                time.sleep(0.5)

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        self.running = False
        self.generation += 1
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def close(self):
        self.stop()
        self.root.destroy()

    def change_channel(self, _event=None):
        try:
            self.first_channel = int(self.channel.get().split(":", 1)[0]) - 1
        except ValueError:
            return
        self.latest = None
        self.draw()
        if self.running:
            self.status.set("Loading selected channels from the latest BESA block…")

    def poll_events(self):
        try:
            while True:
                generation, kind, value = self.events.get_nowait()
                if generation != self.generation:
                    continue
                if kind == "labels":
                    self.channel_box["values"] = [f"{i}: {label}" for i, label in enumerate(value, 1)]
                    if self.first_channel >= len(value):
                        self.first_channel = 0
                    self.channel.set(self.channel_box["values"][self.first_channel])
                elif kind == "chunk":
                    self.latest = append_waveform(self.latest, value)
                    self.last_received = time.time()
                    self.draw()
                    prefix = "Replay" if self.mode.get() == "replay" else "File follow"
                    skipped = f"; skipped {value.skipped_blocks} older blocks" if value.skipped_blocks else ""
                    self.status.set(f"{prefix}: {value.start_s:g}–{value.end_s:g} s; "
                                    f"{len(value.labels)} channels{skipped}")
                elif kind == "status":
                    self.status.set(value)
                elif kind == "finished":
                    self.stop()
                    self.status.set("Replay finished.")
        except queue.Empty:
            pass
        if self.running and self.mode.get() == "follow" and self.latest is not None:
            age = time.time() - self.last_received
            if age > 10:
                self.status.set(f"No new complete BESA block for {age:.0f} s; "
                                f"last displayed {self.latest.start_s:g}–{self.latest.end_s:g} s. "
                                "Check whether the recorder updates this file during acquisition.")
        self.root.after(100, self.poll_events)

    def draw(self):
        self.canvas.delete("all")
        chunk = self.latest
        if chunk is None:
            self.canvas.create_text(30, 30, anchor="nw", text="Waveforms appear here after a complete BESA block is available.",
                                    fill="#59636e")
            return
        width = max(self.canvas.winfo_width(), 500)
        height = max(self.canvas.winfo_height(), 350)
        left, right, top, bottom = 105, width - 20, 25, height - 35
        channels = chunk.data.shape[1]
        lane = (bottom - top) / channels
        self.canvas.create_line(left, top, left, bottom, right, bottom, fill="#7a858f")
        self.canvas.create_text(left, bottom + 8, anchor="nw", text=f"{chunk.start_s:g} s")
        self.canvas.create_text(right, bottom + 8, anchor="ne", text=f"{chunk.end_s:g} s")
        bins = min(max(right - left, 1), 900, len(chunk.data))
        edges = np.linspace(0, len(chunk.data), bins + 1, dtype=int)
        for index, label in enumerate(chunk.labels):
            baseline = top + (index + 0.5) * lane
            self.canvas.create_line(left, baseline, right, baseline, fill="#e3e7eb")
            self.canvas.create_text(left - 8, baseline, anchor="e", text=label, fill="#27333d")
            values = chunk.data[:, index].astype(np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            center = float(np.median(finite))
            scale = float(np.percentile(np.abs(finite - center), 99)) or 1.0
            coords = []
            for j in range(bins):
                part = values[edges[j]:edges[j + 1]]
                part = part[np.isfinite(part)]
                if not len(part):
                    continue
                x = left + (right - left) * (j + 0.5) / bins
                for y_value in (float(part.min()), float(part.max())):
                    y = baseline - (y_value - center) / scale * lane * 0.36
                    coords.extend((x, max(baseline - lane * 0.45, min(baseline + lane * 0.45, y))))
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#1769a4", width=1)


def launch(parent: tk.Tk | None = None):
    if parent is None:
        root = tk.Tk()
        EEGViewer(root)
        root.mainloop()
    else:
        EEGViewer(tk.Toplevel(parent))
