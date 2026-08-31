from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.common import ROOT, activity_from_path, canonicalize_columns, ensure_dirs, estimate_sampling_rate_hz, load_config, write_json


def _tables_from_numbers_parser(path: Path) -> list[pd.DataFrame]:
    try:
        from numbers_parser import Document
    except ImportError as exc:
        raise RuntimeError("numbers-parser is not installed") from exc

    doc = Document(str(path))
    frames: list[pd.DataFrame] = []
    for sheet in doc.sheets:
        for table in sheet.tables:
            rows = table.rows(values_only=True)
            if not rows:
                continue
            header = [str(x) if x is not None else f"col_{i}" for i, x in enumerate(rows[0])]
            data = rows[1:]
            df = pd.DataFrame(data, columns=header)
            df = df.dropna(how="all")
            if len(df):
                frames.append(df)
    if not frames:
        raise RuntimeError(f"No tabular data found in {path}")
    return frames


def _export_with_applescript(path: Path, out_dir: Path) -> Path:
    activity = activity_from_path(path)
    out_path = out_dir / activity / f"{session_id_from_path(path)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = f'''
    tell application "Numbers"
      open POSIX file "{path}"
      set theDoc to front document
      export theDoc to POSIX file "{out_path}" as CSV
      close theDoc saving no
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True)
    return out_path


def session_id_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if stem.endswith("_20hz"):
        stem = stem[: -len("_20hz")]
    return stem


def convert_file(path: Path, out_dir: Path) -> tuple[Path, dict]:
    activity = activity_from_path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / activity / f"{session_id_from_path(path)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session_id = session_id_from_path(path)
    metadata = {"input": str(path), "activity": activity, "session_id": session_id, "method": None, "warnings": []}

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        metadata["method"] = "csv"
    else:
        try:
            frames = _tables_from_numbers_parser(path)
            df = max(frames, key=len)
            metadata["method"] = "numbers-parser"
        except Exception as parser_error:
            metadata["warnings"].append(f"numbers-parser failed: {parser_error}")
            try:
                exported = _export_with_applescript(path, out_dir)
                df = pd.read_csv(exported)
                metadata["method"] = "AppleScript Numbers export"
            except Exception as apple_error:
                raise RuntimeError(
                    f"Could not convert {path}. Install numbers-parser or run on macOS with Numbers.app. "
                    f"numbers-parser error: {parser_error}; AppleScript error: {apple_error}"
                ) from apple_error

    df, assigned = canonicalize_columns(df)
    df["activity"] = activity
    df["session_id"] = session_id
    df["recording_id"] = session_id
    df["source_file"] = path.name
    df.to_csv(out_path, index=False)
    metadata["output"] = str(out_path)
    metadata["columns"] = list(df.columns)
    metadata["detected_columns"] = assigned
    metadata["sampling_rate_hz"] = estimate_sampling_rate_hz(df)
    metadata["rows"] = int(len(df))
    return out_path, metadata


def batch_convert(input_dir: Path, out_dir: Path, cfg: dict) -> list[dict]:
    ensure_dirs()
    files = sorted([*input_dir.glob("*.numbers"), *input_dir.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No .numbers or .csv files found under {input_dir}")
    logs = []
    for path in files:
        session_path, meta = convert_file(path, out_dir)
        meta["output"] = str(session_path)
        logs.append(meta)

    expected = float(cfg["sampling"]["expected_rate_hz"])
    for meta in logs:
        rate = meta.get("sampling_rate_hz")
        if rate is None:
            meta.setdefault("warnings", []).append("Could not estimate sampling rate")
        elif abs(rate - expected) / expected > 0.1:
            meta.setdefault("warnings", []).append(f"Sampling rate {rate:.3f} Hz differs from expected {expected:.3f} Hz")
    write_json(ROOT / "data/raw_csv/conversion_log.json", logs)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "radar_data"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/raw_csv"))
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    args = parser.parse_args()
    logs = batch_convert(Path(args.input_dir), Path(args.output_dir), load_config(args.config))
    for item in logs:
        print(f"{item['activity']}: {item['rows']} rows -> {item['output']}")


if __name__ == "__main__":
    main()
