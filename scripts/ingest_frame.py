# scripts/ingest_frame.py
from __future__ import annotations
import json, math, re, zlib, time, os
from pathlib import Path
from collections import Counter, defaultdict

# ---------- helpers (stdlib only)
TOK = re.compile(r"[A-Za-z0-9_'-]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|•\s+")
SUPPORT = re.compile(r"(^|\s)(therefore|thus|hence|because)\b|^\s*so\b", re.I)
ATTACK  = re.compile(r"\b(however|but|although|nevertheless|nonetheless|on the other hand|contradict|refute)\b", re.I)
ASSUME  = re.compile(r"\b(assume|suppose|hypothesize|let us suppose|let's assume)\b", re.I)

def sentences(t): 
    t=(t or "").strip()
    return [s.strip() for s in re.split(SENT_SPLIT, t) if s.strip()]
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

def structural_entropy(text: str) -> float:
    toks=[w.lower() for w in TOK.findall(text or "")]
    if not toks: return 0.0
    total=len(toks); freq=Counter(toks)
    H=-sum((c/total)*math.log(max(1e-12,c/total)) for c in freq.values())
    Hmax=math.log(len(freq)) if freq else 1.0
    return H/max(1e-12,Hmax)

def sdi(text: str) -> float:
    s=sentences(text)
    if len(s)<2: return 0.0
    bows=[bow(x) for x in s]
    sims=[cosine(bows[i-1], bows[i]) for i in range(1,len(bows))]
    return max(0.0, min(1.0, 1.0 - sum(sims)/len(sims)))

def compute_phi_topo(N,E,D,C,X,S):
    a1,a2,a3,a4,a5 = 0.40,0.20,0.35,0.25,0.30
    raw = a1*math.log(1.0 + E/(N+1.0)) + a2*D - a3*C - a4*X + a5*(S/(X+1.0))
    return sigma(raw)

def compute_phi_sem_proxy(text: str) -> float:
    SDI=sdi(text); CR=compress_ratio(text)
    return max(0.0, min(1.0, 0.6*SDI + 0.4*max(0.0,1.0-CR)))

def rhetorical_rate(text: str) -> float:
    s=sentences(text)
    if not s: return 0.0
    return sum(1 for x in s if (SUPPORT.search(x) or ATTACK.search(x)))/len(s)

def compute_kappa(len_tokens, mu, std, Hs, rate, phi_topo, phi_sem_proxy):
    zL = 0.0 if std<1e-6 else max(-3.0, min(3.0, (len_tokens-mu)/std))
    rho = 0.45*abs(zL) + 0.35*Hs + 0.20*rate
    Sstar = max(1e-3, 0.45*phi_topo + 0.45*phi_sem_proxy + 0.10)
    return sigma(1.4 * rho / Sstar)

# ---------- config & baseline in docs/
REC = Path("docs/receipt.latest.json"); REC.parent.mkdir(parents=True, exist_ok=True)
CFG = Path("docs/guard.config.json")
BASE= Path("docs/baseline.json")

DEFAULT_CFG = {
  "kappa_red": 0.78, "delta_hol_red": 0.22, "max_cycle": 4,
  "kappa_amber": 0.85, "x_amber": 3
}

def load_cfg():
    if CFG.is_file():
        try: 
            data=json.loads(CFG.read_text("utf-8")); 
            return {**DEFAULT_CFG, **data}
        except: pass
    return DEFAULT_CFG

def load_base():
    if BASE.is_file():
        try: return json.loads(BASE.read_text("utf-8"))
        except: pass
    return {"mu_len":400.0,"var_len":400.0,"dhol_mu":0.0,"prev_digest":None}

def save_base(d): BASE.write_text(json.dumps(d,indent=2),encoding="utf-8")

# ---------- main
def build_frame(text: str, prev_digest: dict|None) -> tuple[dict,int,int,int,int,int]:
    s=[x for x in sentences(text)]
    nodes=[]; edges=[]
    for i,t in enumerate(s):
        ntype = "Assumption" if ASSUME.search(t) else ("Evidence" if SUPPORT.search(t) else "Claim")
        nid=f"{ntype[0]}{i+1}"
        nodes.append({"id":nid,"type":ntype,"label":t,"weight":1.0})
        if i==0: continue
        sid=nodes[i-1]["id"]; tid=nodes[i]["id"]
        if SUPPORT.search(t):   edges.append({"src":sid,"dst":tid,"rel":"supports","weight":1.0})
        elif ATTACK.search(t): edges.append({"src":tid,"dst":sid,"rel":"contradicts","weight":1.0})
        elif ASSUME.search(t): edges.append({"src":tid,"dst":tid,"rel":"depends_on","weight":0.8})
    N,E=len(nodes),len(edges)
    S=sum(1 for e in edges if e["rel"]=="supports")
    X=sum(1 for e in edges if e["rel"]=="contradicts")
    # cycles
    g=defaultdict(list); [g[e["src"]].append(e["dst"]) for e in edges]
    color={}; C=0
    def dfs(u):
        nonlocal C
        color[u]=1
        for v in g[u]:
            if color.get(v,0)==0: dfs(v)
            elif color.get(v)==1: C+=1
        color[u]=2
    for n in nodes:
        if color.get(n["id"],0)==0: dfs(n["id"])
    # depth
    indeg=Counter(); [indeg.__setitem__(e["dst"], indeg[e["dst"]]+1) for e in edges]
    roots=[n["id"] for n in nodes if indeg[n["id"]]==0] or ([nodes[0]["id"]] if nodes else [])
    D=0
    def dfsd(u,d):
        nonlocal D; D=max(D,d)
        for v in g[u]: dfsd(v,d+1)
    for r in roots: dfsd(r,1)

    digest={"b0":1,"cycle_plus":C,"x_frontier":X,"s_over_c":(S/max(1,X)),"depth":D}
    keys=("b0","cycle_plus","x_frontier","s_over_c","depth")
    if prev_digest:
        p=[prev_digest.get(k,0) for k in keys]; q=[digest.get(k,0) for k in keys]
        delta = jsd_vec(p,q,1e-9)
    else:
        delta = 0.0
    telem={"phi_sem_proxy":None,"phi_topo":None,"delta_hol":delta,"kappa_eff":None,"cost_tokens":len(TOK.findall(text or ""))}
    return {"digest":digest,"nodes":nodes,"edges":edges,"telem":telem,"raw_text":text}, N,E,S,X,C

def main():
    cfg = load_cfg()
    base= load_base()

    src = Path("docs/input.txt")
    text = src.read_text("utf-8") if src.is_file() else (
        "Claim: Our adapter builds a small reasoning graph. "
        "Because we link evidence with simple markers, structure quality is estimable. "
        "Therefore the guard compares density against structure without chain-of-thought."
    )

    frame, N,E,S,X,C = build_frame(text, base.get("prev_digest"))
    phi_topo = compute_phi_topo(N,E, frame["digest"]["depth"], C, X, S)
    phi_sem  = compute_phi_sem_proxy(text)
    Hs       = structural_entropy(text)
    rate     = rhetorical_rate(text)
    L        = frame["telem"]["cost_tokens"]
    mu, var  = base.get("mu_len",400.0), base.get("var_len",400.0)
    std      = max(1e-6, var)**0.5
    kappa    = compute_kappa(L, mu, std, Hs, rate, phi_topo, phi_sem)

    # smooth holonomy
    dhol_raw = frame["telem"]["delta_hol"]
    dhol_mu  = 0.75*base.get("dhol_mu",0.0) + 0.25*dhol_raw

    frame["telem"].update({"phi_topo":phi_topo,"phi_sem_proxy":phi_sem,"kappa_eff":kappa,"delta_hol":dhol_mu})

    # guard classification (multi-signal)
    status = "green"
    if (dhol_mu > cfg["delta_hol_red"] and kappa > cfg["kappa_red"]) or (frame["digest"]["cycle_plus"] >= cfg["max_cycle"]):
        status = "red"
    elif (kappa > cfg["kappa_amber"]) or (frame["digest"]["x_frontier"] >= cfg["x_amber"]):
        status = "amber"

    # persist baseline
    mu2 = 0.8*mu + 0.2*L
    var2= 0.8*var + 0.2*((L-mu2)**2)
    save_base({"mu_len":mu2,"var_len":var2,"dhol_mu":dhol_mu,"prev_digest":frame["digest"]})

    # merge into receipt
    rec = {}
    if REC.is_file():
        try: rec=json.loads(REC.read_text("utf-8"))
        except: rec={}
    rec["receipt_version"]= rec.get("receipt_version") or "olr/1.2"
    rec.setdefault("attrs",{})["status"]=status
    rec.setdefault("topo",{}).update({
        "kappa": kappa,
        "phi_sem_proxy": phi_sem,
        "rigidity": (phi_topo+phi_sem)/2.0,
        "H": 1.0-kappa
    })
    rec["openline_frame"] = {"digest": frame["digest"], "telem": frame["telem"], "t_logical": int(time.time())}
    rec.setdefault("temporal",{}).setdefault("latest",{}).update({"kappa":kappa,"delta_hol":dhol_mu})
    REC.write_text(json.dumps(rec,indent=2),encoding="utf-8")
    print(f"[ok] frame ingested, status={status}, κ={kappa:.3f}, Δ_hol={dhol_mu:.3f}")

if __name__=="__main__":
    Path("docs").mkdir(parents=True, exist_ok=True)
    main()
