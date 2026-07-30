"""
Run all 10 ablation scripts with a concurrency cap to avoid rate limits.

Usage:
    python run_all_ablations.py [--concurrency N]

    --concurrency N   max simultaneous processes (default: 3)
                      Lower this if you still hit 429s; raise it if you have
                      headroom.  The fallback in model.py handles transient
                      429s, but keeping N ≤ 3 avoids sustained throttling.

Each script gets its own log file under src/ablations/logs/.
Run from anywhere; paths are resolved relative to this file.
"""
import argparse
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

ABLATIONS_DIR = Path(__file__).parent
LOGS_DIR = ABLATIONS_DIR / "logs"
PYTHON = sys.executable
MODEL = "openai"  # maps to gpt-5.1 in model.py

JOBS = [
    # Main dataset (dataset_200_ablation, 200 samples)
    "run_without_C.py",
    "run_without_A.py",
    "run_without_M.py",
    "run_without_web.py",
    "run_without_image.py",
    # AMSNet dataset (amsnet_ablation_100, 100 samples)
    "run_amsnet_without_C.py",
    "run_amsnet_without_A.py",
    "run_amsnet_without_M.py",
    "run_amsnet_without_web.py",
    "run_amsnet_without_image.py",
]


def launch(script: str, timestamp: str):
    script_path = ABLATIONS_DIR / script
    log_path = LOGS_DIR / f"{script.replace('.py', '')}_{timestamp}.log"
    log_f = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        [PYTHON, str(script_path), "--model", MODEL],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(ABLATIONS_DIR),
    )
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] START  [PID {proc.pid:6d}]  {script}  →  {log_path.name}")
    return proc, log_f


def main():
    parser = argparse.ArgumentParser(description="Run all ablations with a concurrency cap.")
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Max number of scripts running at once (default: 3)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated script name substrings to filter (e.g. 'amsnet')",
    )
    args = parser.parse_args()
    concurrency = args.concurrency

    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    jobs = JOBS
    if args.only:
        filters = [f.strip() for f in args.only.split(",")]
        jobs = [j for j in JOBS if any(f in j for f in filters)]

    print(f"Ablation runner  |  model={MODEL}  |  concurrency={concurrency}  |  total={len(jobs)}")
    print(f"Logs: {LOGS_DIR}\n")

    queue = deque(jobs)
    running = {}   # script → (proc, log_f)
    results = {}   # script → exit_code

    while queue or running:
        # Fill slots up to the concurrency cap
        while queue and len(running) < concurrency:
            script = queue.popleft()
            proc, log_f = launch(script, timestamp)
            running[script] = (proc, log_f)

        # Poll running jobs
        time.sleep(10)
        finished = []
        for script, (proc, log_f) in running.items():
            rc = proc.poll()
            if rc is not None:
                finished.append(script)
                results[script] = rc
                log_f.close()
                status = "OK" if rc == 0 else f"FAILED (exit {rc})"
                ts = datetime.now().strftime("%H:%M:%S")
                queued = len(queue)
                print(f"  [{ts}] {status:20s}  {script}  (queued={queued}, running={len(running)-1})")

        for script in finished:
            del running[script]

    # Final summary
    print("\n=== Final Status ===")
    failed = []
    for script in jobs:
        rc = results.get(script, -1)
        mark = "✓" if rc == 0 else "✗"
        print(f"  {mark} {script}  (exit {rc})")
        if rc != 0:
            failed.append(script)

    print()
    if failed:
        print(f"FAILED ({len(failed)}/{len(jobs)}): {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(jobs)} ablations completed successfully.")


if __name__ == "__main__":
    main()
