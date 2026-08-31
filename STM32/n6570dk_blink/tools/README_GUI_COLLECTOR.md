# Radar 20 Hz GUI collector

`radar_collection_gui.py` is the day-to-day desktop collector. Its default
**Teammate-compatible** mode reproduces `DataCollection_Tools/collect_fixed_20hz.py`
for regression comparison. The smarter continuity/nearest-target policy remains
available as a separate diagnostic mode and is not used by compatibility mode.

## Launch

Run from the repository root's STM32 project directory:

```sh
cd radar/STM32
python3 -m pip install pyserial
python3 n6570dk_blink/tools/radar_collection_gui.py
```

Tkinter ships with the standard macOS and Windows Python installers. On Linux,
install the distribution's Tk package if `import tkinter` is unavailable.

Select or type the serial port, keep the default baud rate `115200`, connect,
choose an activity, set duration (default `5` seconds), then start collection.
Choose the target-selection mode before connecting because compatibility mode
also restores the old 10 ms serial timeout. The GUI never writes to the serial
port and does not require the radar RX wire.

Compatibility mode:

- reads only bytes already waiting on the serial port;
- uses the old stateless `0x0A04` scan and its first target only;
- accepts distance 0.3-8.0 m and Z -3.0-2.0 m;
- holds the last accepted target and saves duplicates at scheduled 20 Hz ticks;
- starts after the historical three-second countdown and drops missed ticks;
- uses the actual wall-clock time of each saved row.

## Output

Files are written directly to `n6570dk_blink/training_data/` with no manual renaming:

```text
falling_20260714_160429_20hz.csv
sitting_20260714_132755_20hz.csv
standing_20260714_140537_20hz.csv
walking_20260715_132151_20hz.csv
```

The pattern is `<activity>_<local-start-time:%Y%m%d_%H%M%S>_20hz.csv`.

Every saved file has the existing training schema:

```text
timestamp,activity,frame,cluster_id,x,y,z,dop_idx
```

`cluster_id` is always `1` and exactly one target is used per output row. In
compatibility mode, timestamps contain the actual save times, so normal host
scheduling jitter is retained just as it is in the old datasets. No file is
created if no target passes the old gates. These CSV files require no
post-conversion.

## Live status

The window displays x/y/z, distance, Doppler index, saved frame count, actual
recent write rate, and connection/tracking/collection state. Compatibility
mode deliberately has no freshness timeout: once a target passes the old
gates, it is held until another accepted target replaces it. Other modes retain
the 0.5-second timeout. The raw decoded capture tools remain the multi-target
source of record for diagnostics and research.
