# scripts/cole_demo.py
from scripts.cole_blend import run_demo
from pprint import pprint

if __name__ == "__main__":
    out = run_demo(seed=7, steps=240)
    pprint(out["metrics"])
