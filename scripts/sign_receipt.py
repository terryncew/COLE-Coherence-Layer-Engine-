from __future__ import annotations
import json, os, time
from pathlib import Path
from nacl.signing import SigningKey

IN = Path("docs/receipt.latest.json")
data = json.loads(IN.read_text("utf-8"))

priv_hex = os.environ.get("RECEIPT_SIGNING_KEY", "").strip()
if not priv_hex:
    print("[warn] RECEIPT_SIGNING_KEY not set; skipping signature")
else:
    payload = json.loads(json.dumps({k: data[k] for k in data if k != "sig"}, sort_keys=True, separators=(",",":")))
    msg = json.dumps(payload, sort_keys=True, separators=(",",":")).encode("utf-8")
    sk = SigningKey(bytes.fromhex(priv_hex))
    sig = sk.sign(msg).signature.hex()
    pub = sk.verify_key.encode().hex()
    data["sig"] = {"alg":"Ed25519","key":pub,"ts":int(time.time()),"value":sig}
    print("[ok] signed with pub:", pub[:16]+"…")

IN.write_text(json.dumps(data, indent=2), "utf-8")
print("[ok] wrote", IN)
