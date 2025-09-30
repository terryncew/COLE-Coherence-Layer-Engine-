#!/usr/bin/env python3
# scripts/claude_harness.py
# Text+Vision harness: call Claude (if key), build frame, compute κ/Δhol, emit receipt.

from __future__ import annotations
import os, sys, json, math, re, zlib, time, argparse, base64, mimetypes, urllib.request
from pathlib import Path
from collections import Counter, defaultdict

# ---------- config ----------
WEIGHTS = dict(a1=0.40,a2=0.20,a3=0.35,a4=0.25,a5=0.30,  # φ_topo
               b1=2.0,b2=1.0,                             # φ_sem (cheap)
               rho_zL=0.45,rho_Hs=0.35,rho_r=0.20,        # ρ
               kappa_gamma=1.4, jsd_eps=1e-9)

THRESH = dict(kappa_warn=0.70, kappa_fail_gate=0.70, dhol_fail=0.50,
              phi_floor_warn=0.40, x_frontier_drop_for_deletion=2)

STATE_DIR, ART_DIR = Path(".state"), Path("artifacts")
STATE_DIR.mkdir(parents=True, exist_ok=True); ART_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR/"prev_state.json"
RECEIPT_PATH = ART_DIR/"guard_receipt.json"

# ---------- tiny NLP ----------
TOK = re.compile(r"[A-Za-z0-9_'-]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SUPPORT = re.compile(r"\b(because|since|therefore|thus|hence|so that|so)\b", re.I)
ATTACK  = re.compile(r"\b(however|but|although|nevertheless|nonetheless|on the other hand|contradict|refute)\b", re.I)
ASSUME  = re.compile(r"\b(assume|suppose|hypothesize|let(?:'s)? assume)\b", re.I)
RESOLVE = re.compile(r"\b(resolved|addressed|reconciled|countered|handled|fixed)\b", re.I)

def sents(t:str): return [s.strip() for s in SENT_SPLIT.split(t.strip()) if s.strip()]
def bow(s:str):   return Counter(w.lower() for w in TOK.findall(s))
def cosine(a:Counter,b:Counter)->float:
    if not a or not b: return 0.0
    num=sum(a[k]*b[k] for k in set(a)|set(b))
    den=math.sqrt(sum(v*v for v in a.values()))*math.sqrt(sum(v*v for v in b.values()))
    return (num/den) if den else 0.0
def sigma(x:float)->float:
    if x>=0: z=math.exp(-x); return 1/(1+z)
    z=math.exp(x); return z/(1+z)
def jsd(p,q,eps=1e-9):
    def _n(v): s=sum(v); return [eps if s<=0 else max(eps,x/s) for x in v]
    P,Q=_n(p),_n(q); M=[(a+b)/2 for a,b in zip(P,Q)]
    def _kl(u,v): return sum(ui*math.log(ui/vi) for ui,vi in zip(u,v))
    return 0.5*_kl(P,M)+0.5*_kl(Q,M)

# ---------- frame builder (bounded, heuristic) ----------
def classify(s:str)->str:
    sl=s.lower()
    if ASSUME.search(sl): return "Assumption"
    if SUPPORT.search(sl): return "Evidence"
    if ATTACK.search(sl):  return "Evidence"
    return "Claim"

def count_cycles(nodes,edges)->int:
    g=defaultdict(list)
    for e in edges: g[e["src"]].append(e["dst"])
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
    return cyc

def max_depth(nodes,edges)->int:
    g=defaultdict(list); indeg=Counter()
    for e in edges: g[e["src"]].append(e["dst"]); indeg[e["dst"]]+=1
    roots=[n["id"] for n in nodes if indeg[n["id"]]==0] or ([nodes[0]["id"]] if nodes else [])
    best=0
    def dfs(u,d):
        nonlocal best; best=max(best,d)
        for v in g[u]: dfs(v,d+1)
    for r in roots: dfs(r,1)
    return best

def build_frame(text:str, stream_id:str, t_logical:int, prev_digest:dict|None)->dict:
    ss=sents(text); nodes=[]; edges=[]
    for i,s in enumerate(ss):
        nid=f"{classify(s)[0]}{i+1}"
        nodes.append(dict(id=nid,type=classify(s),label=s,weight=1.0))
    for i,s in enumerate(ss):
        if i==0: continue
        prev=nodes[i-1]["id"]; cur=nodes[i]["id"]
        if SUPPORT.search(s): edges.append(dict(src=prev,dst=cur,rel="supports",weight=1.0))
        elif ATTACK.search(s): edges.append(dict(src=cur,dst=prev,rel="contradicts",weight=1.0))
        elif ASSUME.search(s): edges.append(dict(src=cur,dst=cur,rel="depends_on",weight=0.8))
    N,E=len(nodes),len(edges)
    S=sum(1 for e in edges if e["rel"]=="supports")
    X=sum(1 for e in edges if e["rel"]=="contradicts")
    C=count_cycles(nodes,edges); D=max_depth(nodes,edges)
    digest=dict(b0=1, cycle_plus=C, x_frontier=X, s_over_c=S/max(1,X), depth=D)
    telem=dict(phi_sem=None,phi_topo=None,delta_hol=None,kappa_eff=None,cost_tokens=len(TOK.findall(text)))
    frame=dict(stream_id=stream_id,t_logical=t_logical,nodes=nodes,edges=edges,digest=digest,morphs=[],telem=telem,raw_text=text)
    if prev_digest:
        p=[prev_digest.get(k,0) for k in("b0","cycle_plus","x_frontier","s_over_c","depth")]
        q=[digest.get(k,0)      for k in("b0","cycle_plus","x_frontier","s_over_c","depth")]
        frame["telem"]["delta_hol"]=jsd(p,q,WEIGHTS["jsd_eps"])
    else:
        frame["telem"]["delta_hol"]=0.0
    return frame

# ---------- metrics ----------
def structural_entropy(text:str)->float:
    toks=[w.lower() for w in TOK.findall(text)]
    if not toks: return 0.0
    total=len(toks); freq=Counter(toks)
    H=-sum((c/total)*math.log(max(1e-12,c/total)) for c in freq.values())
    Hmax=math.log(len(freq)+1e-12); return H/max(1e-12,Hmax)

def avg_adj_cos(ss)->float:
    if len(ss)<2: return 0.0
    bows=[bow(x) for x in ss]; sims=[cosine(bows[i-1],bows[i]) for i in range(1,len(bows))]
    return sum(sims)/len(sims)

def compress_ratio(text:str)->float:
    raw=text.encode("utf-8"); comp=zlib.compress(raw,6); return len(comp)/max(1,len(raw))

def phi_topo(N,E,D,C,X,S):
    a1,a2,a3,a4,a5 = WEIGHTS["a1"],WEIGHTS["a2"],WEIGHTS["a3"],WEIGHTS["a4"],WEIGHTS["a5"]
    raw=a1*math.log(1+E/(N+1.0))+a2*D-a3*C-a4*X+a5*(S/(X+1.0))
    return sigma(raw)

def phi_sem(text:str, ss):
    # cheap surrogate: 1) adjacency diversity, 2) mild compression term
    avg=avg_adj_cos(ss)              # 0..1 (higher→more similar)
    CR = compress_ratio(text)        # 0..1+
    b1,b2=WEIGHTS["b1"],WEIGHTS["b2"]
    x=b1*(1.0-avg) - b2*CR
    return max(0.0,min(1.0,sigma(x)))

def compute_kappa(L, mu, std, Hs, rate, phit, phis):
    zL = 0.0 if std<1e-6 else max(-3.0,min(3.0,(L-mu)/std))
    rho = WEIGHTS["rho_zL"]*abs(zL)+WEIGHTS["rho_Hs"]*Hs+WEIGHTS["rho_r"]*rate
    Sstar = max(1e-3, 0.45*phit+0.45*phis+0.10*(1.0))  # C/X term ≈1.0 when low
    return sigma(WEIGHTS["kappa_gamma"]*rho/Sstar), zL

def deletion_suspected(prev_digest,curr_digest,prev_text,curr_text):
    if not prev_digest: return False
    drop = prev_digest.get("x_frontier",0)-curr_digest.get("x_frontier",0)
    if drop >= THRESH["x_frontier_drop_for_deletion"] and not RESOLVE.search((curr_text or "")):
        return True
    return False

# ---------- state ----------
def load_state():
    try: return json.loads(STATE_PATH.read_text("utf-8"))
    except: return {}
def save_state(s:dict): STATE_PATH.write_text(json.dumps(s,indent=2),encoding="utf-8")
def update_ewma(mu,var,x,alpha=0.2):
    mu2=(1-alpha)*mu+alpha*x
    var2=(1-alpha)*var+alpha*((x-mu2)**2)
    return mu2,var2

# ---------- Claude client (text+vision) ----------
def _b64_from_source(src:str)->tuple[str,str]:
    # src may be URL or local path
    if re.match(r"^https?://", src):
        with urllib.request.urlopen(src) as r:
            data=r.read()
            mt=r.headers.get_content_type() or mimetypes.guess_type(src)[0] or "image/jpeg"
    else:
        data=Path(src).read_bytes()
        mt=mimetypes.guess_type(src)[0] or "image/jpeg"
    return base64.b64encode(data).decode("ascii"), mt

def fetch_claude_text(model:str, prompt:str, images:list[str]|None, max_tokens:int=700)->str:
    key=os.getenv("ANTHROPIC_API_KEY")
    if not key: raise RuntimeError("ANTHROPIC_API_KEY not set")
    try:
        import anthropic
    except Exception as e:
        raise RuntimeError("pip install anthropic") from e

    blocks=[]
    # image blocks (if any)
    for src in (images or []):
        try:
            b64,mt=_b64_from_source(src)
            blocks.append({"type":"image","source":{"type":"base64","media_type":mt,"data":b64}})
        except Exception: pass
    # prompt last (so users see instruction after images)
    blocks.append({"type":"text","text": prompt})

    client=anthropic.Anthropic(api_key=key)
    msg=client.messages.create(model=model,max_tokens=max_tokens,temperature=0.2,
                               messages=[{"role":"user","content":blocks}])
    parts=[]
    for c in msg.content:
        if isinstance(c,dict) and c.get("type")=="text": parts.append(c.get("text",""))
        elif hasattr(c,"type") and getattr(c,"type",None)=="text": parts.append(getattr(c,"text",""))
    return "\n\n".join(parts).strip()

# ---------- verdict & receipt ----------
def decide_verdict(kappa,dhol,pt,ps,del_flag):
    warnings=[]; fails=[]
    if pt<THRESH["phi_floor_warn"]: warnings.append("low_phi_topo")
    if ps<THRESH["phi_floor_warn"]: warnings.append("low_phi_sem")
    if kappa>THRESH["kappa_warn"]:  warnings.append("kappa_elevated")
    if (dhol>THRESH["dhol_fail"] or del_flag) and kappa>THRESH["kappa_fail_gate"]:
        fails.append("process_integrity")
    return ("FAIL" if fails else "PASS"), warnings, fails

def write_receipt(frame,metrics,verdict,warnings,fails,images):
    rec={
      "claim":"Guard verdict for one run.",
      "because":[],
      "but":warnings,
      "so":f"verdict={verdict}",
      "temporal":{"latest":{"length_tokens":metrics["L"],"deviation_z":metrics["zL"]}},
      "narrative":{"continuity":{"issues":fails,"notes":[]}},
      "topo":{"kappa":metrics["kappa"],"chi":metrics["phi_sem"],"eps":1.0-metrics["phi_sem"],
              "rigidity":(metrics["phi_topo"]+metrics["phi_sem"])/2.0,
              "D_topo":frame["digest"]["depth"],"H":1.0-metrics["kappa"]},
      "openline_frame":{"digest":frame["digest"],"telem":frame["telem"],"t_logical":frame["t_logical"]},
      "vision":{"images":images or [],"count":len(images or [])},
      "ts":int(time.time())
    }
    RECEIPT_PATH.write_text(json.dumps(rec,indent=2),encoding="utf-8")
    return rec

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser(description="Claude(Text+Vision) → Frame → Guards")
    ap.add_argument("--model",default="claude-3-5-sonnet-20240620")
    ap.add_argument("--prompt",default="Direct technical assessment of the attached visuals; cite two sources if possible.")
    ap.add_argument("--stdin",action="store_true")
    ap.add_argument("--file",type=str)
    ap.add_argument("--image",action="append",help="image URL or path; repeat for multiple")
    args=ap.parse_args()

    # 1) get text (Claude if key; else stdin/file/prompt)
    images=args.image or []
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            text=fetch_claude_text(args.model,args.prompt,images)
        except Exception as e:
            print(f"[warn] Claude failed ({e}); falling back to prompt.")
            text=args.prompt
    elif args.stdin:
        text=sys.stdin.read()
    elif args.file:
        text=Path(args.file).read_text("utf-8")
    else:
        text=args.prompt

    # 2) load state
    st=load_state()
    muL=st.get("mu_len",400.0); varL=st.get("var_len",400.0)
    prev_digest=st.get("prev_digest"); prev_text=st.get("prev_text","")

    # 3) frame
    t_logical=int(st.get("t_logical",0))+1
    frame=build_frame(text,"stream-1",t_logical,prev_digest)

    # 4) metrics
    ss=sents(text)
    N,E=len(frame["nodes"]),len(frame["edges"])
    S=sum(1 for e in frame["edges"] if e["rel"]=="supports")
    X=frame["digest"]["x_frontier"]; C=frame["digest"]["cycle_plus"]; D=frame["digest"]["depth"]
    pt=phi_topo(N,E,D,C,X,S); ps=phi_sem(text,ss)
    Hs=structural_entropy(text)
    rate=0.0 if not ss else sum(1 for s in ss if SUPPORT.search(s) or ATTACK.search(s))/len(ss)
    L=frame["telem"]["cost_tokens"]; stdL=math.sqrt(max(1e-6,varL))
    kappa,zL=compute_kappa(L,muL,stdL,Hs,rate,pt,ps)
    dhol=frame["telem"]["delta_hol"]; del_flag=deletion_suspected(prev_digest,frame["digest"],prev_text,text)
    frame["telem"].update({"phi_topo":pt,"phi_sem":ps,"kappa_eff":kappa})

    # 5) verdict
    verdict,warns,fails=decide_verdict(kappa,dhol,pt,ps,del_flag)

    # 6) persist state
    mu2,var2=update_ewma(muL,varL,L,alpha=0.2)
    save_state(dict(mu_len=mu2,var_len=var2,prev_digest=frame["digest"],prev_text=text,t_logical=t_logical))

    # 7) write receipt
    rec=write_receipt(frame,metrics=dict(L=L,zL=zL,phi_topo=pt,phi_sem=ps,kappa=kappa),verdict=verdict,warnings=warns,fails=fails,images=images)

    # 8) CI line
    print(f"GUARD verdict={verdict} kappa={kappa:.3f} d_hol={dhol:.3f} phi_topo={pt:.3f} phi_sem={ps:.3f} imgs={len(images)} t={t_logical}")

if __name__=="__main__":
    main()
