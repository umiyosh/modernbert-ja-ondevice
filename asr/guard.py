"""Runs a heavy job under a memory watch, so an overnight run cannot exhaust the Mac. On 2026-10-21 a
rescoring job grew to 25 GB beside the TTS and ASR workers; macOS killed system daemons for an hour
and the Mac had to be restarted.

Waits for the heavy-job lock (one GPU job at a time) and for --min-free % free memory, starts the
command in its own process group with an MPS memory cap, and kills that group when its footprint
passes --max-gb, system free memory falls below --stop-free %, or macOS writes a new JetsamEvent
(it has started killing processes). Kills are also appended to guard.log next to this file.
Exits with the command's code, or 137 after a kill. Uses only the standard library.

usage: python3 guard.py [--max-gb 12] [--min-free 40] [--stop-free 20] [--mps 0.3] [--shared] -- cmd ...
       --shared: skip the lock (light workers such as TTS/ASR that run several at once)
"""

import argparse
import fcntl
import glob
import os
import re
import signal
import subprocess
import sys
import time

LOCK = "/tmp/kotobae-heavy.lock"
JETSAM = "/Library/Logs/DiagnosticReports/JetsamEvent-*.ips"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard.log")
POLL = 5
UNITS = {"K": 2**-20, "M": 2**-10, "G": 1.0}


def say(msg, keep=False):
    line = f"{time.strftime('%m-%d %H:%M:%S')} guard[{os.getpid()}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if keep:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def free_pct():
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    if not m:
        raise RuntimeError(f"memory_pressure gave no free percentage: {out!r}")
    return int(m.group(1))


def group_gb(pgid):
    """Footprint of the group in GB. top's MEM counts Metal (MPS) memory, which ps's RSS can miss."""
    out = subprocess.run(["top", "-l", "1", "-stats", "pgrp,mem"], capture_output=True, text=True).stdout
    total = 0.0
    for line in out.splitlines():
        m = re.fullmatch(r"\s*(\d+)\s+([\d.]+)([KMG])[+-]?\s*", line)
        if m and int(m.group(1)) == pgid:
            total += float(m.group(2)) * UNITS[m.group(3)]
    return total


def stop(proc, why):
    say(f"KILL {why}: {' '.join(proc.args)[:200]}", keep=True)
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(wait)
            break
        except subprocess.TimeoutExpired:
            continue
    return 137


def watch(proc, a):
    seen = set(glob.glob(JETSAM))
    peak = 0.0
    while proc.poll() is None:
        gb, free = group_gb(proc.pid), free_pct()
        peak = max(peak, gb)
        if gb > a.max_gb:
            return stop(proc, f"footprint {gb:.1f} GB > {a.max_gb} GB")
        if free < a.stop_free:
            return stop(proc, f"system free {free}% < {a.stop_free}% (job {gb:.1f} GB)")
        new = set(glob.glob(JETSAM)) - seen
        if new:
            return stop(proc, f"macOS is killing processes ({os.path.basename(min(new))}, job {gb:.1f} GB)")
        time.sleep(POLL)
    say(f"done code {proc.returncode}, peak {peak:.1f} GB")
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-gb", type=float, default=12.0)
    p.add_argument("--min-free", type=int, default=40)
    p.add_argument("--stop-free", type=int, default=20)
    p.add_argument("--mps", type=float, default=0.3, help="PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    p.add_argument("--shared", action="store_true")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("no command after --")
    lock = open(LOCK, "w")
    if not a.shared:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            say("waiting for the running heavy job")
            fcntl.flock(lock, fcntl.LOCK_EX)
    while (free := free_pct()) < a.min_free:
        say(f"waiting: free {free}% < {a.min_free}%")
        time.sleep(60)
    # The low watermark must not exceed the high one, or MPS refuses to start.
    env = dict(os.environ, PYTORCH_MPS_HIGH_WATERMARK_RATIO=str(a.mps),
               PYTORCH_MPS_LOW_WATERMARK_RATIO=str(round(a.mps * 0.8, 3)))
    say(f"start (free {free}%, cap {a.max_gb} GB, mps {a.mps}): {' '.join(cmd)[:200]}")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sys.exit(stop(proc, "guard itself was stopped")))
    sys.exit(watch(proc, a))


if __name__ == "__main__":
    main()
