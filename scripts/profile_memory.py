"""Lightweight memory profiler: sample true memory footprint while a workload runs.

On macOS, psutil RSS does NOT include Metal buffer allocations — MLX LLM weights
(mlx_lm) and torch-MPS tensors are invisible to it (verified 2026-07-16: a 2.15GB
MLX allocation moved RSS by +0.01GB). So on macOS we sample phys_footprint via
proc_pid_rusage — the same number Activity Monitor's "Memory" column shows — and
report that as the peak. RSS is still recorded alongside for comparison; on other
platforms footprint falls back to RSS.

Usage:
    python scripts/profile_memory.py -- python scripts/run_indexing.py --passages data/raw/hotpotqa_corpus.json --limit 1000
"""
import ctypes, ctypes.util, subprocess, sys, time, json
from pathlib import Path

try:
    import psutil  # process tree + RSS sampling
except ImportError:
    sys.exit("pip install psutil first")

SAMPLE_INTERVAL_S = 0.5  # 1s sampling missed short-lived peaks (load spikes)

# --- macOS phys_footprint via libproc ---------------------------------------
# rusage_info_v2 layout: 16-byte uuid, then uint64 fields; ri_phys_footprint is
# the 8th uint64 (index 7). The buffer is padded well past v2's real size so a
# kernel writing a larger struct can never overrun it (an undersized struct
# corrupts the heap and segfaults at teardown).
_RUSAGE_INFO_V2 = 2

class _RUsageBuf(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("vals", ctypes.c_uint64 * 64)]

if sys.platform == "darwin":
    _libproc = ctypes.CDLL(ctypes.util.find_library("proc"))

    def phys_footprint(pid: int) -> int:
        buf = _RUsageBuf()
        if _libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(buf)) != 0:
            return 0  # process already gone
        return buf.vals[7]
else:
    def phys_footprint(pid: int) -> int:
        return 0  # non-macOS: footprint column falls back to RSS below


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    cmd = sys.argv[sys.argv.index("--") + 1:]
    proc = subprocess.Popen(cmd)
    p = psutil.Process(proc.pid)
    samples = []
    while proc.poll() is None:
        rss = footprint = 0
        try:
            pids = [proc.pid] + [c.pid for c in p.children(recursive=True)]
        except psutil.NoSuchProcess:
            break
        for pid in pids:
            try:
                rss += psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                continue  # child exited between listing and sampling
            footprint += phys_footprint(pid)
        samples.append({"t": time.time(), "rss_gb": rss / 1e9,
                        "footprint_gb": (footprint or rss) / 1e9})
        time.sleep(SAMPLE_INTERVAL_S)
    out = Path("./results/indexing/memory_profile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(samples, f, indent=2)
    peak_fp = max((s["footprint_gb"] for s in samples), default=0)
    peak_rss = max((s["rss_gb"] for s in samples), default=0)
    print(f"Peak footprint: {peak_fp:.2f} GB (RSS alone: {peak_rss:.2f} GB) "
          f"across {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
