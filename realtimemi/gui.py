"""Small offline desktop UI for a complete BESA or NPZ recording."""

from __future__ import annotations

import csv
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class MIWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RealTimeMI — Local MI Analysis")
        root.geometry("900x620")
        self.events: queue.Queue = queue.Queue()
        self.source = tk.StringVar()
        self.output = tk.StringVar(value=str((Path.cwd() / "mi-output").resolve()))
        self.start = tk.StringVar(value="0")
        self.duration = tk.StringVar(value="300")
        self.status = tk.StringVar(value="Select a BESA recording or NPZ demo.")
        self.channel = tk.StringVar(value="1")
        self.rows: list[list[str]] = []

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Recording (.besa or .npz)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.source).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse…", command=self.pick_source).grid(row=1, column=1)
        ttk.Label(frame, text="Results folder").grid(row=2, column=0, sticky="w", pady=(14, 0))
        ttk.Entry(frame, textvariable=self.output).grid(row=3, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse…", command=self.pick_output).grid(row=3, column=1)
        options = ttk.Frame(frame)
        options.grid(row=4, column=0, sticky="w", pady=14)
        ttk.Label(options, text="Start (s)").pack(side="left")
        ttk.Entry(options, textvariable=self.start, width=8).pack(side="left", padx=(6, 18))
        ttk.Label(options, text="Duration (s)").pack(side="left")
        ttk.Entry(options, textvariable=self.duration, width=8).pack(side="left", padx=6)
        self.run_button = ttk.Button(frame, text="Calculate MI", command=self.run)
        self.run_button.grid(row=4, column=1, sticky="e")
        ttk.Label(frame, textvariable=self.status).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))

        results = ttk.LabelFrame(frame, text="MI result", padding=10)
        results.grid(row=6, column=0, columnspan=2, sticky="nsew")
        ttk.Label(results, text="Channel").grid(row=0, column=0, sticky="w")
        self.channel_box = ttk.Combobox(results, textvariable=self.channel, width=9, state="readonly")
        self.channel_box.grid(row=0, column=1, sticky="w", padx=8)
        self.channel_box.bind("<<ComboboxSelected>>", lambda _: self.draw())
        self.canvas = tk.Canvas(results, height=260, background="white", highlightthickness=1,
                                highlightbackground="#d0d0d0")
        self.canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=10)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.table = ttk.Treeview(results, columns=("start", "end", "mi"), show="headings", height=8)
        for key, title, width in (("start", "Start (s)", 100), ("end", "End (s)", 100),
                                  ("mi", "MI", 140)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center")
        self.table.grid(row=2, column=0, columnspan=2, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(6, weight=1)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(1, weight=1)
        results.rowconfigure(2, weight=1)
        root.after(100, self.poll)

    def pick_source(self):
        path = filedialog.askopenfilename(filetypes=[("Supported recordings", "*.besa *.npz"),
                                                 ("All files", "*")])
        if path:
            self.source.set(path)

    def pick_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output.set(path)

    def run(self):
        try:
            start = float(self.start.get())
            duration = float(self.duration.get())
            if start < 0 or duration < 30:
                raise ValueError("Start must be nonnegative and duration at least 30 seconds")
            source, output = self.source.get(), self.output.get()
            if not source or not output:
                raise ValueError("Select a BESA or NPZ file and results folder")
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc))
            return
        self.run_button.configure(state="disabled")
        self.status.set("Starting analysis…")

        def worker():
            try:
                from realtimemi.pipeline import analyze_besa
                info = analyze_besa(source, output, start_seconds=start,
                                    duration_seconds=duration,
                                    progress=lambda message: self.events.put(("status", message)))
                self.events.put(("done", info))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status.set(value)
                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self.status.set("Analysis failed")
                    messagebox.showerror("Analysis error", value)
                elif kind == "done":
                    self.run_button.configure(state="normal")
                    self.status.set(f"Done in {value['elapsed_s']['result_ready']:.2f} s — {value['csv_path']}")
                    with open(value["csv_path"], newline="", encoding="utf-8") as stream:
                        self.rows = list(csv.reader(stream))[1:]
                    self.channel_box["values"] = [str(i) for i in range(1, value["source_channels"] + 1)]
                    self.channel.set("1")
                    self.draw()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def draw(self):
        self.canvas.delete("all")
        for item in self.table.get_children():
            self.table.delete(item)
        if not self.rows:
            return
        column = int(self.channel.get()) + 1
        points = []
        for row in self.rows:
            value = float(row[column])
            self.table.insert("", "end", values=(row[0], row[1], "undefined" if value != value else f"{value:.6g}"))
            if value == value:
                points.append((float(row[0]), value))
        if not points:
            return
        width = max(self.canvas.winfo_width(), 500)
        height = max(self.canvas.winfo_height(), 260)
        left, top, right, bottom = 50, 20, width - 20, height - 35
        self.canvas.create_line(left, top, left, bottom, right, bottom, fill="#777")
        self.canvas.create_text(8, top, anchor="nw", text="MI")
        self.canvas.create_text(right, bottom + 8, anchor="ne", text="Time (s)")
        y_max = max(y for _, y in points)
        if y_max <= 0:
            y_max = 1e-6
        x_max = max(x for x, _ in points)
        if x_max <= 0:
            x_max = 1
        coords = []
        for x, y in points:
            coords.extend((left + (right - left) * x / x_max,
                           bottom - (bottom - top) * y / y_max))
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill="#0866a9", width=2)
        for x, y in zip(coords[::2], coords[1::2]):
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#0866a9", outline="")
        self.canvas.create_text(left - 4, top, anchor="e", text=f"{y_max:.3g}")


def launch():
    root = tk.Tk()
    MIWindow(root)
    root.mainloop()
