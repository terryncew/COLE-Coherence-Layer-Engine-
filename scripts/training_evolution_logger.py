# scripts/training_evolution_logger.py
# Populate docs/receipt.latest.json.training_evolution from common train/eval logs.
# Safe if logs are missing. All stdlib.
from __future__ import annotations
from pathlib import Path
import json, os, hashlib, math

REC = Path("docs/receipt.latest.json")
LOG_DIR = Path(os.getenv("TRAIN_LOG_DIR","logs/train"))
EVAL_PATH = Path(os.getenv("EVAL_LOG","logs/eval/results.jsonl"))
HP_PATH = Path(os.getenv("HP_SCHEDULE_JSON","logs/hparams.json"))

def _read_jsonl(p: Path, keystep="step", limit=200000):
    out=[]
    if p.is_file():
        for i, line in enumerate(p.read_text("utf-8").splitlines()):
            s = line.strip()
            if not s: continue
            try:
                j=json.loads(s)
                if keystep in j: out.append(j)
            except Exception:
                pass
            if i>=limit: break
    return out

def _hp_digest(path: Path)->str|None:
    if not path.is_file(): return None
    try:
        raw = path.read_bytes()
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return None

def _ckpt_lineage(rows):
    if not rows: return []
    out=[]
    for r in rows:
        out.append({
            "step": r.get("step"),
            "ckpt": r.get("checkpoint") or r.get("path") or f"ckpt_{r.get('step')}",
            "loss": r.get("loss"),
            "eval": r.get("eval"),
            "sparsity": r.get("sparsity"),
            "head_hash": r.get("head_hash")
        })
    # keep sparse lineage (every K)
    K= max(1, int(os.getenv("CKPT_STRIDE","50")))
    out = [x for i,x in enumerate(sorted(out, key=lambda z: (z["step"] or 0))) if i%K==0 or i==len(out)-1]
    return out

def _capability_events(eval_rows):
    out=[]
    thresholds = {"f1":0.80, "acc":0.80, "exact":0.60}
    first_seen={}
    for r in eval_rows:
        step=r.get("step")
        for k,thr in thresholds.items():
            if k in r:
                v=r.get(k)
                if isinstance(v,(int,float)) and v >= thr and k not in first_seen:
                    first_seen[k]=(step, v)
    for name,(step,score) in first_seen.items():
        out.append({"name":name,"step":step,"score":score})
    return out

def _loss_shape(train_rows):
    xs=sorted([(r.get("step"), r.get("loss")) for r in train_rows if r.get("loss") is not None and r.get("step") is not None])
    if len(xs)<3: return {"largest_jump":0.0,"inflections":0,"ewma_slope":0.0}
    jumps=[abs(xs[i][1]-xs[i-1][1]) for i in range(1,len(xs))]
    largest=max(jumps) if jumps else 0.0
    inf=0
    for i in range(2,len(xs)):
        d1=xs[i-1][1]-xs[i-2][1]
        d2=xs[i][1]-xs[i-1][1]
        if d1==0 or d2==0: continue
        if (d1>0 and d2<0) or (d1<0 and d2>0): inf+=1
    alpha=0.2; m=xs[0][1]
    for _,y in xs[1:]: m=(1-alpha)*m+alpha*y
    span = max(1,(xs[-1][0]-xs[0][0]))
    slope=(m - xs[0][1]) / span
    return {"largest_jump": float(largest), "inflections": int(inf), "ewma_slope": float(slope)}

def _grad_stats(grad_rows):
    vals=[r.get("grad_norm") for r in grad_rows if isinstance(r.get("grad_norm"), (int,float))]
    if not vals: return {"mean":0.0,"std":0.0,"max":0.0}
    mean=sum(vals)/len(vals)
    var=sum((v-mean)**2 for v in vals)/len(vals)
    return {"mean":float(mean),"std":float(math.sqrt(var)),"max":float(max(vals))}

def _ensure_receipt() -> dict:
    if REC.is_file():
        try: return json.loads(REC.read_text("utf-8"))
        except Exception: pass
    return {"receipt_version":"olr/1.2"}

def main():
    seed = os.getenv("SEED") or os.getenv("PYTHONHASHSEED") or "unknown"

    # read logs (all optional)
    train_rows = _read_jsonl(LOG_DIR/"train.jsonl")
    ckpt_rows  = _read_jsonl(LOG_DIR/"checkpoints.jsonl")
    grad_rows  = _read_jsonl(LOG_DIR/"grads.jsonl")
    eval_rows  = _read_jsonl(Path(EVAL_PATH))

    lineage = _ckpt_lineage(ckpt_rows or train_rows)
    hp_dig  = _hp_digest(Path(HP_PATH))
    cap     = _capability_events(eval_rows)
    shape   = _loss_shape(train_rows)
    grads   = _grad_stats(grad_rows)

    rec = _ensure_receipt()
    te = rec.setdefault("training_evolution", {})
    te["seed"]= seed
    te["checkpoint_lineage"]= lineage
    te["hp_schedule_digest"]= hp_dig
    te["capability_events"]= cap
    te["loss_trajectory_shape"]= shape
    te["grad_norm_stats"]= grads
    te["causal_link_hint"]= bool(te.get("causal_link_hint", False))

    # curriculum guard: large early jump + non-negative ewma slope → nudge amber
    try:
      earliest = lineage[0]["step"] if lineage else None
      latest   = lineage[-1]["step"] if lineage else None
      early_jump = shape["largest_jump"] if (earliest is not None and latest is not None and earliest < (latest//4)) else 0.0
    except Exception:
      early_jump = 0.0
    if early_jump and shape.get("ewma_slope", 0.0) > -1e-3:
        attrs = rec.setdefault("attrs", {})
        attrs["status"] = "amber"
        rec.setdefault("but", []).append("curriculum_jump")

    REC.parent.mkdir(parents=True, exist_ok=True)
    REC.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("[ok] training_evolution logged → receipt (seed:", seed, ")")

if __name__=="__main__":
    main()
