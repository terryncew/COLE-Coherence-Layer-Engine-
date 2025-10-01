# scripts/validate_suite.py
from __future__ import annotations
import json, math, re, csv, os, pathlib, zlib
from collections import Counter, defaultdict
P = pathlib.Path

# --- tiny text->frame + metrics (matches your ingest logic) ---
TOK = re.compile(r"[A-Za-z0-9_'-]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SUPPORT = re.compile(r"\b(because|since|therefore|thus|hence|so that|so)\b", re.I)
ATTACK  = re.compile(r"\b(however|but|although|nevertheless|nonetheless|on the other hand|contradict|refute)\b", re.I)
ASSUME  = re.compile(r"\b(assume|suppose|hypothesize|let us suppose|let's assume)\b", re.I)

def sentences(t): 
    t=(t or "").strip()
    return [s.strip() for s in SENT_SPLIT.split(t) if s.strip()]

def bow(s): return Counter(w.lower() for w in TOK.findall(s or ""))

def cosine(a: Counter, b: Counter) -> float:
    if not a or not b: return 0.0
    ks=set(a)|set(b); num=sum(a[k]*b[k] for k in ks)
    den=(sum(v*v for v in a.values())**0.5)*(sum(v*v for v in b.values())**0.5)
    return (num/den) if den else 0.0

def sigma(x: float) -> float:
    if x >= 0: 
        z = math.exp(-x); return 1/(1+z)
    z = math.exp(x); return z/(1+z)

def compress_ratio(text: str) -> float:
    raw=(text or "").encode("utf-8")
    if not raw: return 1.0
    return len(zlib.compress(raw, 6))/max(1,len(raw))

def jsd_vec(p, q, eps=1e-9):
    def _n(v):
        s=sum(v); 
        return [eps]*len(v) if s<=0 else [max(eps, x/s) for x in v]
    P,Q=_n(p),_n(q); M=[(pi+qi)/2 for pi,qi in zip(P,Q)]
    def _kl(u,v): return sum(ui*math.log(ui/vi) for ui,vi in zip(u,v))
    return 0.5*_kl(P,M)+0.5*_kl(Q,M)

def sdi(text: str) -> float:
    s=sentences(text)
    if len(s)<2: return 0.0
    bows=[bow(x) for x in s]
    sims=[cosine(bows[i-1], bows[i]) for i in range(1,len(bows))]
    return max(0.0, min(1.0, 1.0 - sum(sims)/len(sims)))

def structural_entropy(text: str) -> float:
    toks=[w.lower() for w in TOK.findall(text or "")]
    if not toks: return 0.0
    total=len(toks); freq=Counter(toks)
    H=-sum((c/total)*math.log(max(1e-12,c/total)) for c in freq.values())
    Hmax=math.log(len(freq)) if freq else 1.0
    return H/max(1e-12,Hmax)

def rhetorical_rate(text: str) -> float:
    s=sentences(text)
    if not s: return 0.0
    return sum(1 for x in s if (SUPPORT.search(x) or ATTACK.search(x)))/len(s)

def compute_phi_topo(N,E,D,C,X,S):
    a1,a2,a3,a4,a5 = 0.40,0.20,0.35,0.25,0.30
    raw = a1*math.log(1.0 + E/(N+1.0)) + a2*D - a3*C - a4*X + a5*(S/(X+1.0))
    return sigma(raw)

def compute_phi_sem(text: str) -> float:
    SDI=sdi(text); CR=compress_ratio(text)
    return max(0.0, min(1.0, 0.6*SDI + 0.4*max(0.0,1.0-CR)))

def build_frame(text: str):
    s=sentences(text); nodes=[]; edges=[]
    for i, t in enumerate(s):
        ntype = "Assumption" if ASSUME.search(t) else ("Evidence" if SUPPORT.search(t) else "Claim")
        nid = f"{ntype[0]}{i+1}"
        nodes.append({"id":nid,"type":ntype,"label":t})
        if i==0: continue
        sid = nodes[i-1]["id"]; tid = nodes[i]["id"]
        if SUPPORT.search(t):   edges.append({"src":sid,"dst":tid,"rel":"supports"})
        elif ATTACK.search(t): edges.append({"src":tid,"dst":sid,"rel":"contradicts"})
        elif ASSUME.search(t): edges.append({"src":tid,"dst":tid,"rel":"depends_on"})
    N,E=len(nodes),len(edges)
    S=sum(1 for e in edges if e["rel"]=="supports")
    X=sum(1 for e in edges if e["rel"]=="contradicts")
    # cycles (simple DFS)
    g=defaultdict(list); [g[e["src"]].append(e["dst"]) for e in edges]
    color={}; cyc=0
    def dfs(u):
        nonlocal cyc
        color[u]=1
        for v in g[u]:
            if color.get(v,0)==0: dfs(v)
            elif color.get(v)==1: cyc+=1
        color[u]=2
    for n in nodes:
        if color.get(n["id"],0)==0: dfs(n["id"])
    C=cyc
    # depth
    indeg=Counter(); [indeg.__setitem__(e["dst"], indeg.get(e["dst"],0)+1) for e in edges]
    roots=[n["id"] for n in nodes if indeg[n["id"]]==0] or ([nodes[0]["id"]] if nodes else [])
    best=0
    def ddfs(u,d):
        nonlocal best; best=max(best,d)
        for v in g[u]: ddfs(v,d+1)
    for r in roots: ddfs(r,1)
    D=best
    digest={"b0":1,"cycle_plus":C,"x_frontier":X,"s_over_c":(S/max(1,X)),"depth":D}
    return {"nodes":nodes,"edges":edges,"digest":digest}

def compute_kappa(text: str, phi_topo: float, phi_sem: float, length_mu: float, length_std: float):
    L=len(TOK.findall(text or ""))
    zL = 0.0 if length_std<1e-6 else max(-3.0, min(3.0, (L-length_mu)/length_std))
    Hs = structural_entropy(text)          # 0..1
    rate = rhetorical_rate(text)           # 0..1
    rho = 0.45*abs(zL) + 0.35*Hs + 0.20*rate
    Sstar = max(1e-3, 0.45*phi_topo + 0.45*phi_sem + 0.10)
    return sigma(2.5 * (rho / Sstar - 1.0)), L

def delta_hol(prev_digest, cur_digest):
    if not prev_digest: return 0.0
    keys=("b0","cycle_plus","x_frontier","s_over_c","depth")
    p=[prev_digest.get(k,0) for k in keys]; q=[cur_digest.get(k,0) for k in keys]
    return jsd_vec(p,q,1e-9)

# --- evaluation ---
def evaluate(dataset_path="docs/validation/examples.jsonl", out_dir="docs/validation"):
    out = P(out_dir); out.mkdir(parents=True, exist_ok=True)
    xs = [json.loads(line) for line in P(dataset_path).read_text("utf-8").splitlines() if line.strip()]
    # compute corpus length stats
    lens=[len(TOK.findall(x.get("text",""))) for x in xs]
    mu=sum(lens)/max(1,len(lens)); var=sum((l-mu)**2 for l in lens)/max(1,len(lens)); std=max(1e-6, var)**0.5

    rows=[]; prev_digest = None
    for ex in xs:
        text = ex["text"]; label = ex.get("label","unknown")   # good|bad|circular|deletion
        frame = build_frame(text)
        N=len(frame["nodes"]); E=len(frame["edges"])
        S=sum(1 for e in frame["edges"] if e["rel"]=="supports")
        X=frame["digest"]["x_frontier"]; C=frame["digest"]["cycle_plus"]; D=frame["digest"]["depth"]
        phi_topo = compute_phi_topo(N,E,D,C,X,S)
        phi_sem  = compute_phi_sem(text)
        kappa, L = compute_kappa(text, phi_topo, phi_sem, mu, std)
        dhol = delta_hol(prev_digest, frame["digest"])
        prev_digest = frame["digest"]
        rows.append({"id": ex.get("id"), "label": label, "kappa": round(kappa,4), "delta_hol": round(dhol,4),
                     "cycle_plus": C, "x_frontier": X, "phi_topo": round(phi_topo,3), "phi_sem": round(phi_sem,3)})

    # thresholds (v0 sensible defaults)
    T_K_RED = 0.85
    T_DHOL  = 0.30

    # task buckets and simple scoring
    # circular → cycle_plus>0 ; deletion → delta_hol>=T_DHOL with x_frontier drop vs previous (we approximate by dhol alone here)
    preds=[]
    for i, r in enumerate(rows):
        pred_bad = (r["kappa"]>=T_K_RED) or (r["delta_hol"]>=T_DHOL) or (r["cycle_plus"]>0)
        preds.append("bad" if pred_bad else "good")

    # compute coarse metrics
    gold = [("bad" if ("bad" in r["label"] or "circular" in r["label"] or "deletion" in r["label"]) else "good") for r in rows]
    tp=sum(1 for g,p in zip(gold,preds) if g=="bad" and p=="bad")
    tn=sum(1 for g,p in zip(gold,preds) if g=="good" and p=="good")
    fp=sum(1 for g,p in zip(gold,preds) if g=="good" and p=="bad")
    fn=sum(1 for g,p in zip(gold,preds) if g=="bad" and p=="good")
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)

    # write CSV
    csv_path = out/"results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())+["pred","gold"])
        w.writeheader()
        for r,p,g in zip(rows,preds,gold):
            rr=r.copy(); rr["pred"]=p; rr["gold"]=g; w.writerow(rr)

    # write HTML summary (Pages-friendly)
    html = f"""<html><head><meta charset="utf-8"><title>Validation</title>
    <style>body{{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;padding:24px;}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
    .card{{border:1px solid #e5e7eb;border-radius:10px;padding:16px}}</style></head><body>
    <h2>OpenLine / COLE — Validation (n={len(rows)})</h2>
    <div class="grid">
      <div class="card"><h3>Thresholds</h3><pre>κ ≥ {T_K_RED} → stress (red)
Δ_hol ≥ {T_DHOL} → path drift
cycle_plus > 0 → circular</pre></div>
      <div class="card"><h3>Summary</h3>
      <pre>TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}
Precision: {prec:.2f}   Recall: {rec:.2f}</pre></div>
    </div>
    <p>CSV: <a href="results.csv">results.csv</a></p>
    </body></html>"""
    (out/"index.html").write_text(html, encoding="utf-8")
    print(f"[ok] wrote {csv_path} and {out/'index.html'}")

if __name__=="__main__":
    evaluate()
