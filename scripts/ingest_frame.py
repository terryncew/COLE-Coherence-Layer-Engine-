# scripts/ingest_frame.py
from __future__ import annotations
import json, math, re, zlib, time
from pathlib import Path
from collections import Counter, defaultdict

# -------- regex & utils (stdlib only)
TOK = re.compile(r"[A-Za-z0-9_'-]+")
SPLIT = re.compile(r"(?<=[.!?])\s+|•\s+")
SUPPORT = re.compile(r"(^|\s)(therefore|thus|hence|because)\b|^\s*so\b", re.I)
ATTACK  = re.compile(r"\b(however|but|although|nevertheless|nonetheless|on the other hand|contradict|refute)\b", re.I)
ASSUME  = re.compile(r"\b(assume|suppose|hypothesize|let us suppose|let's assume)\b", re.I)

def sents(t): return [s.strip() for s in re.split(SPLIT, (t or "").strip()) if s.strip()]
def bow(s): return Counter(w.lower() for w in TOK.findall(s or ""))

def cos(a: Counter, b: Counter) -> float:
    if not a or not b: return 0.0
    ks=set(a)|set(b); num=sum(a[k]*b[k] for k in ks)
    den=(sum(v*v for v in a.values())**0.5)*(sum(v*v for v in b.values())**0.5)
    return (num/den) if den else 0.0

def sigmoid(x: float) -> float:
    if x >= 0: z = math.exp(-x); return 1/(1+z)
    z = math.exp(x); return z/(1+z)

def compress_ratio(text: str) -> float:
    raw=(text or "").encode("utf-8")
    if not raw: return 1.0
    return len(zlib.compress(raw, 6))/max(1,len(raw))

def jsd(p, q, eps=1e-9):
    def _n(v):
        s=sum(v); 
        return [eps]*len(v) if s<=0 else [max(eps,x/s) for x in v]
    P,Q=_n(p),_n(q); M=[(pi+qi)/2 for pi,qi in zip(P,Q)]
    def _kl(u,v): return sum(ui*math.log(ui/vi) for ui,vi in zip(u,v))
    return 0.5*_kl(P,M)+0.5*_kl(Q,M)

def entropy_0_1(text: str) -> float:
    toks=[w.lower() for w in TOK.findall(text or "")]
    if not toks: return 0.0
    total=len(toks); freq=Counter(toks)
    H=-sum((c/total)*math.log(max(1e-12,c/total)) for c in freq.values())
    Hmax=math.log(len(freq)) if freq else 1.0
    return H/max(1e-12,Hmax)

def rate_0_1(text: str) -> float:
    ss=sents(text)
    if not ss: return 0.0
    hits=sum(1 for s in ss if (SUPPORT.search(s) or ATTACK.search(s)))
    r=hits/len(ss)
    return min(r/0.2, 1.0)  # soft cap

def sdi(text: str) -> float:
    ss=sents(text)
    if len(ss)<2: return 0.0
    bows=[bow(x) for x in ss]
    sims=[cos(bows[i-1], bows[i]) for i in range(1,len(bows))]
    return max(0.0, min(1.0, 1.0 - sum(sims)/len(sims)))

def phi_topo(N,E,D,C,X,S):
    a1,a2,a3,a4,a5 = 0.40,0.20,0.35,0.25,0.30
    raw = a1*math.log(1.0 + E/(N+1.0)) + a2*D - a3*C - a4*X + a5*(S/(X+1.0))
    return sigmoid(raw)

def phi_sem_proxy(text: str) -> float:
    SDI=sdi(text); CR=compress_ratio(text)
    return max(0.0, min(1.0, 0.6*SDI + 0.4*max(0.0,1.0-CR)))

def compute_kappa(L, mu, std, Hs_0_1, rate_0_1_, phi_topo_, phi_sem_, gamma=2.5):
    zL = 0.0 if std < 1e-6 else max(-3.0, min(3.0, (L-mu)/std))
    rho = 0.45*abs(zL) + 0.35*Hs_0_1 + 0.20*rate_0_1_
    Sstar = max(1e-3, 0.45*phi_topo_ + 0.45*phi_sem_ + 0.10)
    return sigmoid(gamma * (rho/Sstar - 1.0))

# -------- files
REC = Path("docs/receipt.latest.json"); REC.parent.mkdir(parents=True, exist_ok=True)
CFG = Path("docs/guard.config.json")
BASE= Path("docs/baseline.json")
INPUT = Path("docs/input.txt")

DEFAULT_CFG = {
  "kappa_red": 0.85,
  "delta_hol_red": 0.35,
  "max_cycle": 4,
  "kappa_amber": 0.75,
  "x_amber": 3
}

def load_cfg():
    if CFG.is_file():
        try: return {**DEFAULT_CFG, **json.loads(CFG.read_text("utf-8"))}
        except: pass
    return DEFAULT_CFG

def load_base():
    if BASE.is_file():
        try: return json.loads(BASE.read_text("utf-8"))
        except: pass
    return {"mu_len":400.0,"var_len":400.0,"dhol_mu":0.0,"prev_digest":None}

def save_base(d): BASE.write_text(json.dumps(d,indent=2),encoding="utf-8")

def build_frame(text: str, prev_digest: dict|None):
    ss=sents(text); nodes=[]; edges=[]
    for i,t in enumerate(ss):
        ntype = "Assumption" if ASSUME.search(t) else ("Evidence" if SUPPORT.search(t) else "Claim")
        nid=f"{ntype[0]}{i+1}"; nodes.append({"id":nid,"type":ntype,"label":t,"weight":1.0})
        if i==0: continue
        sid=nodes[i-1]["id"]; tid=nodes[i]["id"]
        if SUPPORT.search(t):   edges.append({"src":sid,"dst":tid,"rel":"supports","weight":1.0})
        elif ATTACK.search(t): edges.append({"src":tid,"dst":sid,"rel":"contradicts","weight":1.0})
        elif ASSUME.search(t): edges.append({"src":tid,"dst":tid,"rel":"depends_on","weight":0.8})
    N,E=len(nodes),len(edges)
    S=sum(1 for e in edges if e["rel"]=="supports")
    X=sum(1 for e in edges if e["rel"]=="contradicts")

    # cycles + depth
    g=defaultdict(list); [g[e["src"]].append(e["dst"]) for e in edges]
    color={}; C=0
    def dfs(u):
        nonlocal C
        color[u]=1
        for v in g[u]:
            if color.get(v,0)==0: dfs(v)
            elif color.get(v)==1: C += 1
        color[u]=2
    for n in nodes:
        if color.get(n["id"],0)==0: dfs(n["id"])
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
        dhol_raw = jsd(p,q,1e-9)
    else:
        dhol_raw = 0.0

    telem={"phi_sem_proxy":None,"phi_topo":None,"delta_hol":dhol_raw,"kappa_eff":None,"cost_tokens":len(TOK.findall(text or ""))}
    return {"digest":digest,"nodes":nodes,"edges":edges,"telem":telem,"raw_text":text}, (N,E,S,X,C)

def main():
    cfg  = load_cfg()
    base = load_base()

    text = INPUT.read_text("utf-8") if INPUT.is_file() else (
        "Claim: The adapter builds a small reasoning graph. "
        "Because we link evidence with markers, structure quality is estimable. "
        "Therefore density vs structure is checkable."
    )

    frame,(N,E,S,X,C) = build_frame(text, base.get("prev_digest"))
    phiT = phi_topo(N,E, frame["digest"]["depth"], C, X, S)
    phiS = phi_sem_proxy(text)
    Hs   = entropy_0_1(text)
    rate = rate_0_1(text)
    L    = frame["telem"]["cost_tokens"]
    mu,var = base.get("mu_len",400.0), base.get("var_len",400.0)
    std  = max(1e-6, var)**0.5
    kappa= compute_kappa(L, mu, std, Hs, rate, phiT, phiS, gamma=2.5)

    # EWMA holonomy
    dhol_sm = 0.75*base.get("dhol_mu",0.0) + 0.25*frame["telem"]["delta_hol"]
    frame["telem"].update({"phi_topo":phiT,"phi_sem_proxy":phiS,"kappa_eff":kappa,"delta_hol":dhol_sm})

    # guard rule (dual-trigger for RED)
    status = "green"
    if (dhol_sm > cfg["delta_hol_red"] and kappa > cfg["kappa_red"]) or (frame["digest"]["cycle_plus"] >= cfg["max_cycle"]):
        status = "red"
    elif (kappa > cfg["kappa_amber"]) or (frame["digest"]["x_frontier"] >= cfg["x_amber"]):
        status = "amber"

    # persist baselines
    mu2 = 0.8*mu + 0.2*L
    var2= 0.8*var + 0.2*((L-mu2)**2)
    save_base({"mu_len":mu2,"var_len":var2,"dhol_mu":dhol_sm,"prev_digest":frame["digest"]})

    # merge into receipt
    REC.parent.mkdir(parents=True, exist_ok=True)
    rec = {}
    if REC.is_file():
        try: rec=json.loads(REC.read_text("utf-8"))
        except: rec={}
    rec["receipt_version"]= rec.get("receipt_version") or "olr/1.2"
    rec.setdefault("attrs",{})["status"]=status
    rec.setdefault("topo",{}).update({
        "kappa": kappa, "phi_sem_proxy": phiS, "rigidity": (phiT+phiS)/2.0, "H": 1.0-kappa
    })
    rec["openline_frame"] = {"digest": frame["digest"], "telem": frame["telem"], "t_logical": int(time.time())}
    rec.setdefault("temporal",{}).setdefault("latest",{}).update({"kappa":kappa,"delta_hol":dhol_sm})
    REC.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"GUARD VERDICT status={status} kappa={kappa:.3f} d_hol={dhol_sm:.3f} phi_topo={phiT:.3f} phi_sem_proxy={phiS:.3f}")

if __name__=="__main__":
    main()
