"""Command-line and desktop entry point."""

import argparse
import json
from multiprocessing import freeze_support


def main():
    parser = argparse.ArgumentParser(description="Local BESA or NPZ modulation-index analysis")
    parser.add_argument("--input", help="Complete BESA recording or 1 kHz NPZ demo")
    parser.add_argument("--output-dir", default="mi-output")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hash-input", action="store_true", help="Record source SHA-256 after result")
    parser.add_argument("--viewer", action="store_true", help="Open the BESA waveform viewer")
    args = parser.parse_args()
    if args.viewer:
        from realtimemi.viewer_gui import launch
        launch()
        return
    if not args.input:
        from realtimemi.gui import launch
        launch()
        return
    from realtimemi.pipeline import analyze_besa
    info = analyze_besa(args.input, args.output_dir, start_seconds=args.start,
                        duration_seconds=args.duration, workers=args.workers,
                        hash_input=args.hash_input, progress=print)
    print(json.dumps({"result_ready_s": info["elapsed_s"]["result_ready"],
                      "csv": info["csv_path"], "metadata": info["metadata_path"]}))


if __name__ == "__main__":
    freeze_support()
    main()
