# scripts/label_receipt.py
# Usage:
#   python scripts/label_receipt.py docs/receipt.latest.json pass "sounds; supported with evidence"
# Verdict ∈ {pass, borderline, fail}. Adds top-level "human_eval" block (human_eval.v1).

import json, sys, pathlib, datetime as dt

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    if len(sys.argv) < 3:
        print("usage: python scripts/label_receipt.py <docs/receipt*.json> <pass|borderline|fail> [note...]")
        sys.exit(2)

    p = ROOT / sys.argv[1]
    if not p.is_file():
        print(f"not found: {p}")
        sys.exit(2)

    verdict = sys.argv[2].strip().lower()
    if verdict not in {"pass","borderline","fail"}:
        print("verdict must be one of: pass | borderline | fail")
        sys.exit(2)

    note = " ".join(sys.argv[3:]).strip()

    data = json.loads(p.read_text(encoding="utf-8"))
    data["human_eval"] = {
        "verdict": verdict,
        "reasons": [],  # optional: ["unsupported_claims","contradiction","policy_violation","style_only","other"]
        "quality": {"accuracy":0, "coherence":0, "helpfulness":0},
        "safety":  {"privacy":0, "toxicity":0, "policy_compliance":0},
        "notes": note,
        "annotator": {"role":"author","anon_id":""},
        "iat": dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "schema": "human_eval.v1"
    }

    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"label_receipt: added human_eval to {p}")

if __name__ == "__main__":
    main()
