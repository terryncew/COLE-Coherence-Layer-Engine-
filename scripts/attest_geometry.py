from pathlib import Path
import json, sys, os

REC = Path("docs/receipt.latest.json")
REQ = os.getenv("REQUIRE_GEOMETRY_ATTESTATION", "1")  # set "0" to allow missing

if not REC.exists():
    print("[err] docs/receipt.latest.json missing"); sys.exit(2)

j = json.loads(REC.read_text("utf-8"))
rows = j.get("geometry_attestation") or []

caps = {"spectral_max": 2.0, "orthogonality_error": 0.08, "lipschitz_budget_used": 0.80}
breaches = []

if REQ == "1" and not rows:
    breaches.append("missing geometry_attestation[]")

for row in rows:
    for k, cap in caps.items():
        try:
            v = float(row.get(k, 0.0))
            if v > cap:
                breaches.append(f'{row.get("layer","?")}.{k}={v:.3f} > {cap:.2f}')
        except Exception:
            continue

if breaches:
    j.setdefault("emergency", {})
    j["emergency"]["quench_mode"] = "geometry"
    j["emergency"]["quench_reason"] = "; ".join(breaches)
    j.setdefault("but", [])
    msg = f"Geometry breach → {', '.join(breaches)}"
    if msg not in j["but"]:
        j["but"].insert(0, msg)
    j.setdefault("attrs", {}); j["attrs"]["status"] = "red"
    REC.write_text(json.dumps(j, indent=2), encoding="utf-8")
    print("[fail] geometry breach → QUENCH:", "; ".join(breaches))
    sys.exit(2)

print("[ok] geometry within caps")
