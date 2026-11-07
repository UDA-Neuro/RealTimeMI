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

from realtimemi.viewer import BesaBlockSource, WaveformChunk, WaveformPlayer
from realtimemi.live_mi import LiveMIPlanner, LiveMIResult, compute_live_mi_window
from realtimemi.live_mi_store import LiveMIResultStore


class EEGViewer:
    def __init__(self, root: tk.Tk | tk.Toplevel, output_dir: str | Path | None = None):
        self.root = root
        root.title("RealTimeMI — BESA EEG Viewer")
        root.geometry("1100x700")
        self.path = tk.StringVar()
        self.mode = tk.StringVar(value="follow")
        self.channel = tk.StringVar(value="1")
        self.status = tk.StringVar(value="Choose a BESA file, then start the viewer.")
        self.position = tk.StringVar(value="Display --")
        self.events: queue.Queue = queue.Queue()
        self.running = False
        self.active_mode = "follow"
        self.generation = 0
        self.first_channel = 0
        self.latest: WaveformChunk | None = None
        self.player = WaveformPlayer(window_s=10)
        self.source_finished = False
        self.last_frame = time.monotonic()
        self.last_received = 0.0
        self.mi_status = tk.StringVar(value="MI: waiting for 30 seconds of signal.")
        self.mi_planner = LiveMIPlanner()
        self.mi_busy = False
        self.mi_compute_lock = threading.Lock()
        self.mi_epoch = 0
        self.mi_results: dict[int, dict[int, tuple[float, bool]]] = {}
        self.mi_channel_labels: list[str] = []
        self.mi_selected_channel: int | None = None
        self.mi_output_dir = Path(output_dir) if output_dir else Path.cwd() / "mi-output"
        self.mi_store: LiveMIResultStore | None = None
        self.mi_save_status = tk.StringVar(value=f"MI results: {self.mi_output_dir}")

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
        ttk.Label(controls, textvariable=self.position).pack(side="left", padx=(24, 0))
        ttk.Label(frame, textvariable=self.status, wraplength=1000).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.canvas = tk.Canvas(frame, background="white", highlightthickness=1,
                                highlightbackground="#c5cbd0")
        self.canvas.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _: self.draw())
        mi_panel = ttk.LabelFrame(frame, text="Modulation Index · 3–4 Hz phase × 80–250 Hz amplitude", padding=8)
        mi_panel.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        self.mi_table = ttk.Treeview(mi_panel, columns=("channel", "mi", "window", "state"),
                                     show="headings", height=7, selectmode="browse")
        for key, title, width in (("channel", "Channel", 80), ("mi", "Latest MI", 90),
                                  ("window", "Window (s)", 110), ("state", "Status", 90)):
            self.mi_table.heading(key, text=title)
            self.mi_table.column(key, width=width, anchor="center")
        self.mi_table.grid(row=0, column=0, sticky="nsew")
        self.mi_table.bind("<<TreeviewSelect>>", self.select_mi_channel)
        self.mi_canvas = tk.Canvas(mi_panel, height=170, background="white", highlightthickness=1,
                                   highlightbackground="#c5cbd0")
        self.mi_canvas.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.mi_canvas.bind("<Configure>", lambda _: self.draw_mi())
        ttk.Label(mi_panel, textvariable=self.mi_status).grid(row=1, column=0, columnspan=2,
                                                                sticky="w", pady=(5, 0))
        ttk.Label(mi_panel, textvariable=self.mi_save_status).grid(row=2, column=0, columnspan=2,
                                                                    sticky="w")
        mi_panel.columnconfigure(1, weight=1)
        mi_panel.rowconfigure(0, weight=1)
        ttk.Label(frame, text="Waveform display: 1 kHz antialias resampling; each channel autoscaled in ADC counts.").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(7, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=3)
        frame.rowconfigure(5, weight=2)
        root.after(100, self.poll_events)
        root.after(100, self.tick)
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
        self.active_mode = self.mode.get()
        self.reset_mi()
        self.source_finished = False
        self.player.clear()
        self.latest = None
        self.position.set("Display --")
        self.draw()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        mode = self.active_mode
        self.status.set("Waiting for a complete BESA data block…")

        def worker():
            source = BesaBlockSource(path, replay=mode == "replay")
            announced = False
            last_error = None
            while self.running and generation == self.generation:
                try:
                    selected_channel = self.first_channel
                    chunk = source.poll(selected_channel)
                    if source.labels and not announced:
                        self.events.put((generation, "labels", source.labels))
                        announced = True
                    if chunk is not None:
                        self.events.put((generation, "chunk", (selected_channel, chunk)))
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
        self.player.clear()
        self.reset_mi()
        self.position.set("Display --")
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
                    self.mi_channel_labels = value
                    self.channel_box["values"] = [f"{i}: {label}" for i, label in enumerate(value, 1)]
                    if self.first_channel >= len(value):
                        self.first_channel = 0
                    self.channel.set(self.channel_box["values"][self.first_channel])
                    self.refresh_mi_table()
                elif kind == "chunk":
                    selected_channel, value = value
                    if selected_channel != self.first_channel:
                        continue
                    first_chunk = self.player.available is None
                    self.player.push(value)
                    if first_chunk:
                        self.last_frame = time.monotonic()
                    self.latest = self.player.visible()
                    self.last_received = time.time()
                    self.draw()
                    prefix = "Replay" if self.active_mode == "replay" else "File follow"
                    skipped = f"; skipped {value.skipped_blocks} older blocks" if value.skipped_blocks else ""
                    self.status.set(f"{prefix}: {value.start_s:g}–{value.end_s:g} s; "
                                    f"{len(value.labels)} channels{skipped}")
                elif kind == "status":
                    self.status.set(value)
                elif kind == "finished":
                    self.source_finished = True
                elif kind == "mi_done":
                    epoch, result = value
                    if epoch == self.mi_epoch:
                        self.mi_busy = False
                        self.mi_planner.complete(result)
                        self.record_mi_result(result)
                elif kind == "mi_error":
                    epoch, start, error = value
                    if epoch == self.mi_epoch:
                        self.mi_busy = False
                        self.mi_planner.failed(start)
                        self.mi_status.set(f"MI {start}–{start + 30} s: {error}; retrying")
        except queue.Empty:
            pass
        if self.running and self.active_mode == "follow" and self.latest is not None:
            age = time.time() - self.last_received
            if age > 10:
                self.status.set(f"No new complete BESA block for {age:.0f} s; "
                                f"last displayed {self.latest.start_s:g}–{self.latest.end_s:g} s. "
                                "Check whether the recorder updates this file during acquisition.")
        self.root.after(100, self.poll_events)

    def tick(self):
        now = time.monotonic()
        elapsed = now - self.last_frame
        self.last_frame = now
        if self.running and self.player.advance(elapsed):
            self.latest = self.player.visible()
            self.draw()
        if self.player.available is not None and self.player.cursor_s is not None:
            self.position.set(f"Display {self.player.cursor_s:.1f} s / committed "
                              f"{self.player.available.end_s:.1f} s")
            self.schedule_mi()
        if (self.running and self.source_finished and not self.mi_busy and
                not self.mi_planner.pending and
                (self.player.caught_up or self.player.available is None)):
            self.stop()
            self.status.set("Replay finished.")
        self.root.after(100, self.tick)

    def reset_mi(self):
        self.mi_epoch += 1
        self.mi_planner = LiveMIPlanner(replay=self.active_mode == "replay")
        self.mi_busy = False
        self.mi_results.clear()
        self.mi_selected_channel = None
        self.mi_store = None
        self.mi_save_status.set(f"MI results: {self.mi_output_dir}")
        self.mi_status.set("MI: waiting for 30 seconds of signal.")
        self.refresh_mi_table()
        self.draw_mi()

    def schedule_mi(self):
        if not self.running or not self.mi_channel_labels or self.mi_busy:
            return
        available = self.player.available
        if available is None:
            return
        self.mi_planner.observe(self.player.cursor_s or available.start_s, available.end_s)
        start = self.mi_planner.pop()
        if start is None:
            return
        channels = tuple(range(self.first_channel,
                               min(self.first_channel + 8, len(self.mi_channel_labels))))
        generation, epoch = self.generation, self.mi_epoch
        path = Path(self.path.get())
        self.mi_busy = True
        self.mi_status.set(f"MI: calculating {start}–{start + 30} s on GPU…")

        def worker():
            try:
                with self.mi_compute_lock:
                    if generation != self.generation or epoch != self.mi_epoch:
                        return
                    result = compute_live_mi_window(path, start, channels)
                self.events.put((generation, "mi_done", (epoch, result)))
            except Exception as exc:
                self.events.put((generation, "mi_error",
                                 (epoch, start, f"{type(exc).__name__}: {exc}")))

        threading.Thread(target=worker, daemon=True).start()

    def record_mi_result(self, result: LiveMIResult):
        for channel, value in zip(result.channels, result.values):
            self.mi_results.setdefault(channel, {})[result.start_s] = (
                float(value), result.final_context)
        status = "final" if result.final_context else "provisional; awaiting filter context"
        self.mi_status.set(f"MI: {result.start_s}–{result.end_s} s, {status}, "
                           f"computed in {result.elapsed_s:.2f} s")
        try:
            if self.mi_store is None:
                self.mi_store = LiveMIResultStore(self.mi_output_dir, self.mi_channel_labels)
            self.mi_store.append(result)
            self.mi_save_status.set(f"MI CSV: {self.mi_store.csv_path}")
        except OSError as exc:
            self.mi_save_status.set(f"MI could not be saved: {exc}")
        self.refresh_mi_table()
        self.draw_mi()

    def refresh_mi_table(self):
        if not hasattr(self, "mi_table"):
            return
        for item in self.mi_table.get_children():
            self.mi_table.delete(item)
        for channel in range(self.first_channel,
                             min(self.first_channel + 8, len(self.mi_channel_labels))):
            history = self.mi_results.get(channel, {})
            label = self.mi_channel_labels[channel]
            if history:
                start = max(history)
                value, final = history[start]
                mi_text = "undefined" if not np.isfinite(value) else f"{value:.6g}"
                values = (label, mi_text, f"{start}–{start + 30}",
                          "final" if final else "provisional")
            else:
                values = (label, "—", "—", "waiting")
            self.mi_table.insert("", "end", iid=str(channel), values=values)
        if self.mi_selected_channel is None and self.mi_table.get_children():
            self.mi_selected_channel = int(self.mi_table.get_children()[0])
        if self.mi_selected_channel is not None and self.mi_table.exists(str(self.mi_selected_channel)):
            self.mi_table.selection_set(str(self.mi_selected_channel))

    def select_mi_channel(self, _event=None):
        selection = self.mi_table.selection()
        if selection:
            self.mi_selected_channel = int(selection[0])
            self.draw_mi()

    def draw_mi(self):
        if not hasattr(self, "mi_canvas"):
            return
        self.mi_canvas.delete("all")
        channel = self.mi_selected_channel
        if channel is None:
            self.mi_canvas.create_text(15, 15, anchor="nw", text="MI trend appears after 30 s.",
                                       fill="#59636e")
            return
        history = self.mi_results.get(channel, {})
        points = [(start + 30, value) for start, (value, _) in sorted(history.items())
                  if np.isfinite(value)]
        if not points:
            self.mi_canvas.create_text(15, 15, anchor="nw", text="MI trend appears after 30 s.",
                                       fill="#59636e")
            return
        width = max(self.mi_canvas.winfo_width(), 300)
        height = max(self.mi_canvas.winfo_height(), 150)
        left, right, top, bottom = 48, width - 15, 18, height - 28
        self.mi_canvas.create_line(left, top, left, bottom, right, bottom, fill="#7a858f")
        label = self.mi_channel_labels[channel] if channel < len(self.mi_channel_labels) else str(channel + 1)
        self.mi_canvas.create_text(right, top, anchor="ne", text=f"{label} MI", fill="#43505d")
        x_first = max(0, points[0][0] - 15)
        x_last = max(x_first + 30, points[-1][0])
        y_max = max(0.001, max(value for _, value in points) * 1.1)
        coords = []
        for end, value in points:
            x = left + (right - left) * (end - x_first) / (x_last - x_first)
            y = bottom - (bottom - top) * value / y_max
            coords.extend((x, y))
        if len(coords) >= 4:
            self.mi_canvas.create_line(*coords, fill="#b04a91", width=2)
        for x, y in zip(coords[::2], coords[1::2]):
            self.mi_canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                       fill="#b04a91", outline="")
        self.mi_canvas.create_text(left, bottom + 5, anchor="nw", text=f"{x_first:g} s")
        self.mi_canvas.create_text(right, bottom + 5, anchor="ne", text=f"{x_last:g} s")
        self.mi_canvas.create_text(left - 5, top, anchor="e", text=f"{y_max:.3g}")

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
        origin = self.player.origin_s if self.player.origin_s is not None else chunk.start_s
        window_start = max(origin, chunk.end_s - self.player.window_s)
        window_end = window_start + self.player.window_s
        self.canvas.create_line(left, top, left, bottom, right, bottom, fill="#7a858f")
        self.canvas.create_text(left, bottom + 8, anchor="nw", text=f"{window_start:.1f} s")
        self.canvas.create_text(right, bottom + 8, anchor="ne", text=f"{window_end:.1f} s")
        cursor_x = left + (right - left) * (chunk.end_s - window_start) / self.player.window_s
        self.canvas.create_line(cursor_x, top, cursor_x, bottom, fill="#b7c3ce", dash=(3, 3))
        bins = min(max(right - left, 1), 900, len(chunk.data))
        step = max(1, (len(chunk.data) + bins - 1) // bins)
        starts = np.arange(0, len(chunk.data), step)
        sample_centers = np.minimum(starts + step / 2, len(chunk.data) - 1)
        x = left + (right - left) * (
            chunk.start_s + sample_centers / 1000 - window_start) / self.player.window_s
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
            clean = np.where(np.isfinite(values), values, center)
            low = np.minimum.reduceat(clean, starts)
            high = np.maximum.reduceat(clean, starts)
            y_low = baseline - (low - center) / scale * lane * 0.36
            y_high = baseline - (high - center) / scale * lane * 0.36
            y_low = np.clip(y_low, baseline - lane * 0.45, baseline + lane * 0.45)
            y_high = np.clip(y_high, baseline - lane * 0.45, baseline + lane * 0.45)
            coords = np.column_stack((x, y_low, x, y_high)).ravel().tolist()
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#1769a4", width=1)


def launch(parent: tk.Tk | None = None, output_dir: str | Path | None = None):
    if parent is None:
        root = tk.Tk()
        EEGViewer(root, output_dir)
        root.mainloop()
    else:
        EEGViewer(tk.Toplevel(parent), output_dir)
