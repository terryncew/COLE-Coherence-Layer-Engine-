# scripts/ingest_14L.py  (OLR/1.4L — lean, first-run safe, short-answer smart)
from __future__ import annotations
import json, math, re, time, zlib
from pathlib import Path
from collections import Counter, defaultdict

# --------- tiny NLP ----------
TOK = re.compile(r"[A-Za-z0-9_'-]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SUPPORT = re.compile(r"\b(because|since|therefore|thus|hence|so that|so|according to|study|studies|evidence|source|citation|as shown)\b", re.I)
ATTACK  = re.compile(r"\b(however|but|although|nevertheless|nonetheless|on the other hand|contradict|refute)\b", re.I)
ASSUME  = re.compile(r"\b(assume|suppose|hypothesize|let us suppose|let's assume)\b", re.I)
RESOLVE = re.compile(r"\b(resolved|addressed|reconciled|countered|handled|fixed|mitigated)\b", re.I)
CITEPAT = re.compile(r"\[[0-9]{1,3}\]|https?://|doi:\S+", re.I)

def sentences(t:str):
    t=(t or "").strip()
    return [s.strip() for s in SENT_SPLIT.split(t) if s.strip()]

def bow(s:str) -> Counter:
    return Counter(w.lower() for w in TOK.findall(s or ""))

def cosine(a:Counter, b:Counter)->float:
    if not a or not b: return 0.0
    ks=set(a)|set(b)
    num=sum(a[k]*b[k] for k in ks)
    den=(sum(v*v for v in a.values())**0.5)*(sum(v*v for v in b.values())**0.5)
    return (num/den) if den else 0.0

def compress_ratio(text:str)->float:
    raw=(text or "").encode("utf-8")
    if not raw: return 1.0
    return len(zlib.compress(raw, 6))/max(1,len(raw))

def sdi(text:str)->float:
    s=sentences(text)
    if len(s)<2: return 0.0
    bows=[bow(x) for x in s]
    sims=[cosine(bows[i-1], bows[i]) for i in range(1,len(bows))]
    return max(0.0, min(1.0, 1.0 - sum(sims)/len(sims)))

def sigma(x:float)->float:
    if x>=0:
        z=math.exp(-x); return 1/(1+z)
    z=math.exp(x); return z/(1+z)

def jsd_vec(p, q, eps=1e-9):
    def _n(v):
        s=sum(v)
        return [eps]*len(v) if s<=0 else [max(eps, x/s) for x in v]
    P,Q=_n(p),_n(q); M=[(pi+qi)/2 for pi,qi in zip(P,Q)]
    def _kl(u,v): return sum(ui*math.log(ui/vi) for ui,vi in zip(u,v))
    return 0.5*_kl(P,M)+0.5*_kl(Q,M)

# --------- graph + metrics ----------
def classify_sentence(s:str)->str:
    sl=s.lower()
    if ASSUME.search(sl): return "Assumption"
    if SUPPORT.search(sl): return "Evidence"
    if ATTACK.search(sl):  return "Claim"     # still a claim, but with contrast marker
    return "Claim"

def build_graph(text:str):
    s=sentences(text); nodes=[]; edges=[]
    for i,t in enumerate(s):
        ntype=classify_sentence(t); nid=f"{ntype[0]}{i+1}"
        nodes.append({"id":nid,"type":ntype,"label":t,"weight":1.0})
        if i==0: continue
        sid=nodes[i-1]["id"]; tid=nodes[i]["id"]
        if SUPPORT.search(t):   edges.append({"src":sid,"dst":tid,"rel":"supports","weight":1.0})
        elif ATTACK.search(t):  edges.append({"src":tid,"dst":sid,"rel":"contradicts","weight":1.0})
        elif ASSUME.search(t):  edges.append({"src":tid,"dst":tid,"rel":"depends_on","weight":0.8})
    return nodes, edges

def count_cycles(nodes, edges)->int:
    g=defaultdict(list)
    for e in edges: g[e["src"]].append(e["dst"])
    color={}; cycles=0
    def dfs(u):
        nonlocal cycles
        color[u]=1
        for v in g[u]:
            if color.get(v,0)==0: dfs(v)
            elif color.get(v)==1: cycles+=1
        color[u]=2
    for n in nodes:
        if color.get(n["id"],0)==0: dfs(n["id"])
    return cycles

def max_depth(nodes, edges)->int:
    g=defaultdict(list); indeg=Counter()
    for e in edges: g[e["src"]].append(e["dst"]); indeg[e["dst"]]+=1
    roots=[n["id"] for n in nodes if indeg[n["id"]]==0] or ([nodes[0]["id"]] if nodes else [])
    best=0
    def dfs(u,d):
        nonlocal best; best=max(best,d)
        for v in g[u]: dfs(v,d+1)
    for r in roots: dfs(r,1)
    return best

def compute_phi_topo(N,E,D,C,X,S):
    a1,a2,a3,a4,a5 = 0.40,0.20,0.35,0.25,0.30
    raw = a1*math.log(1.0 + E/(N+1.0)) + a2*D - a3*C - a4*X + a5*(S/(X+1.0))
    return sigma(raw)

def evidence_strength(text:str)->float:
    s=sentences(text)
    if not s: return 0.0
    hits=0
    for t in s:
        if SUPPORT.search(t) or CITEPAT.search(t): hits+=1
    return hits/len(s)

def phi_sem_v2(text:str)->float:
    SDI=sdi(text); CR=compress_ratio(text); ES=evidence_strength(text)
    return max(0.0, min(1.0, 0.5*SDI + 0.3*max(0.0,1.0-CR) + 0.2*ES))

def rhetorical_rate(text:str)->float:
    s=sentences(text)
    if not s: return 0.0
    return sum(1 for t in s if (SUPPORT.search(t) or ATTACK.search(t)))/len(s)

def unsupported_claim_ratio(text:str)->float:
    s=sentences(text)
    if not s: return 0.0
    claims=[t for t in s if classify_sentence(t)=="Claim" and len(TOK.findall(t))>=5]
    if not claims: return 0.0
    supported_window=set()
    for i,t in enumerate(s):
       
