# scripts/ci_guard_harness.py
from __future__ import annotations
import json, sys, subprocess, shlex
from pathlib import Path

REC = Path("docs/receipt.latest.json")

def run(cmd: str):
    print(f"[run] {cmd}")
    p = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    sys.stdout.write(p.stdout); sys.stderr.write(p.stderr)
    if p.returncode != 0:
        print(f"[warn] command exited {p.returncode}")
    return p.returncode

def main():
    # run the ingest step (same as your Pages job uses)
    rc = run("python scripts/ingest_frame.py")
    if not REC.is_file():
        print("VERDICT: NO-RECEIPT")
        sys.exit(2)
    r = json.loads(REC.read_text("utf-8"))
    status = (r.get("attrs") or {}).get("status", "unknown")
    telem  = ((r.get("openline_frame") or {}).get("telem") or {})
    kappa  = telem.get("kappa_eff", r.get("topo",{}).get("kappa"))
    dhol   = telem.get("delta_hol", (r.get("temporal",{}).get("latest") or {}).get("delta_hol"))
    print(f"VERDICT: {status.upper()}  (kappa={kappa:.3f}  delta_hol={dhol:.3f})")
    # exit non-zero only on RED so PR checks can fail if you want
    sys.exit(1 if status == "red" else 0)

if __name__ == "__main__":
    main()
