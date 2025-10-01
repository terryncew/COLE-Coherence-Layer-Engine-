# scripts/ingest_14L.py  (OLR/1.4L, stdlib-only, first-run safe)
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

def sentences(t: str):
    t = (t or "").strip()
    return [s.strip() for s in SENT_SPLIT.split(t) if s.strip()]

def bow(s: str) -> Counter:
    return Counter(w.lower() for w in TOK.findall(s or ""))

def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    ks = set(a) | set(b)
    num = sum(a[k] * b[k] for k in ks)
    den = (sum(v * v for v in a.values()) ** 0.5) * (sum(v * v for v in b.values()) ** 0.5)
    return (num / den) if den else 0.0

def compress_ratio(text: str) -> float:
    raw = (text or "").encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / max(1, len(raw))

def sdi(text: str) -> float:
    s = sentences(text)
    if len(s) < 2:
        return 0.0
    bows = [bow(x) for x in s]
    sims = [cosine(bows[i - 1], bows[i]) for i in range(1, len(bows))]
    return max(0.0, min(1.0, 1.0 - sum(sims) / len(sims)))

def sigma(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)

def jsd_vec(p, q, eps=1e-9):
    def _n(v):
        s = sum(v)
        return [eps] * len(v) if s <= 0 else [max(eps, x / s) for x in v]
    P, Q = _n(p), _n(q)
    M = [(pi + qi) / 2 for pi, qi in zip(P, Q)]
    def _kl(u, v):
        return sum(ui * math.log(ui / vi) for ui, vi in zip(u, v))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

# --------- graph + metrics ----------
def classify_sentence(s: str) -> str:
    sl = s.lower()
    if ASSUME.search(sl):
        return "Assumption"
    if SUPPORT.search(sl):
        return "Evidence"
    if ATTACK.search(sl):
        return "Claim"      # claim with contrast marker still counts as a claim
    return "Claim"

def build_graph(text: str):
    s = sentences(text)
    nodes, edges = [], []
    for i, t in enumerate(s):
        ntype = classify_sentence(t)
        nid = f"{ntype[0]}{i+1}"
        nodes.append({"id": nid, "type": ntype, "label": t, "weight": 1.0})
        if i == 0:
            continue
        sid = nodes[i - 1]["id"]
        tid = nodes[i]["id"]
        if SUPPORT.search(t):
            edges.append({"src": sid, "dst": tid, "rel": "supports", "weight": 1.0})
        elif ATTACK.search(t):
            edges.append({"src": tid, "dst": sid, "rel": "contradicts", "weight": 1.0})
        elif ASSUME.search(t):
            edges.append({"src": tid, "dst": tid, "rel": "depends_on", "weight": 0.8})
    return nodes, edges

def count_cycles(nodes, edges) -> int:
    g = defaultdict(list)
    for e in edges:
        g[e["src"]].append(e["dst"])
    color = {}
    cycles = 0
    def dfs(u):
        nonlocal cycles
        color[u] = 1
        for v in g[u]:
            if color.get(v, 0) == 0:
                dfs(v)
            elif color.get(v) == 1:
                cycles += 1
        color[u] = 2
    for n in nodes:
        if color.get(n["id"], 0) == 0:
            dfs(n["id"])
    return cycles

def max_depth(nodes, edges) -> int:
    g = defaultdict(list)
    indeg = Counter()
    for e in edges:
        g[e["src"]].append(e["dst"])
        indeg[e["dst"]] += 1
    roots = [n["id"] for n in nodes if indeg[n["id"]] == 0] or ([nodes[0]["id"]] if nodes else [])
    best = 0
    def dfs(u, d):
        nonlocal best
        best = max(best, d)
        for v in g[u]:
            dfs(v, d + 1)
    for r in roots:
        dfs(r, 1)
    return best

def compute_phi_topo(N, E, D, C, X, S):
    a1, a2, a3, a4, a5 = 0.40, 0.20, 0.35, 0.25, 0.30
    raw = a1 * math.log(1.0 + E / (N + 1.0)) + a2 * D - a3 * C - a4 * X + a5 * (S / (X + 1.0))
    return sigma(raw)

def evidence_strength(text: str) -> float:
    s = sentences(text)
    if not s:
        return 0.0
    hits = 0
    for t in s:
        if SUPPORT.search(t) or CITEPAT.search(t):
            hits += 1
    return hits / len(s)

def phi_sem_v2(text: str) -> float:
    SDI = sdi(text)
    CR = compress_ratio(text)
    ES = evidence_strength(text)
    return max(0.0, min(1.0, 0.5 * SDI + 0.3 * max(0.0, 1.0 - CR) + 0.2 * ES))

def rhetorical_rate(text: str) -> float:
    s = sentences(text)
    if not s:
        return 0.0
    return sum(1 for t in s if (SUPPORT.search(t) or ATTACK.search(t))) / len(s)

def unsupported_claim_ratio(text: str) -> float:
    s = sentences(text)
    if not s:
        return 0.0
    claims = [t for t in s if classify_sentence(t) == "Claim" and len(TOK.findall(t)) >= 5]
    if not claims:
        return 0.0
    supported_window = set()
    for i, t in enumerate(s):
        if SUPPORT.search(t) or CITEPAT.search(t):
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(s):
                    supported_window.add(j)
    unsupported = sum(1 for i, t in enumerate(s) if t in claims and i not in supported_window)
    return unsupported / max(1, len(claims))

def compute_kappa(length_tokens: int, mu_len: float, var_len: float,
                  Hs: float, rate: float, UCR: float, phi_topo: float, phi_sem: float) -> float:
    std = max(1e-6, var_len) ** 0.5
    zL = 0.0 if std < 1e-6 else max(-3.0, min(3.0, (length_tokens - mu_len) / std))
    rho = 0.40 * abs(zL) + 0.30 * Hs + 0.15 * rate + 0.15 * UCR
    Sstar = max(1e-3, 0.50 * phi_topo + 0.40 * phi_sem + 0.10)
    return sigma(2.2 * (rho / Sstar - 1.0))

def structural_entropy(text: str) -> float:
    toks = [w.lower() for w in TOK.findall(text or "")]
    if not toks:
        return 0.0
    total = len(toks)
    freq = Counter(toks)
    H = -sum((c / total) * math.log(max(1e-12, c / total)) for c in freq.values())
    Hmax = math.log(len(freq)) if freq else 1.0
    return H / max(1e-12, Hmax)

# --------- state, IO, verdict ----------
STATE = Path(".state/olr14l.json"); STATE.parent.mkdir(parents=True, exist_ok=True)
REC   = Path("docs/receipt.latest.json"); REC.parent.mkdir(parents=True, exist_ok=True)
CFG   = Path("docs/guard.14l.json")

DEFAULT_CFG = {
    "tau_k": 0.75,
    "tau_hol": 0.35,
    "ucr_min": 0.40,
    "es_min": 0.25
}

def load_state():
    if STATE.is_file():
        try:
            return json.loads(STATE.read_text("utf-8"))
        except:
            pass
    # first-run seed so Δhol has a stable previous vector
    return {
        "mu_len": 400.0,
        "var_len": 400.0,
        "prev_digest": {"b0": 1, "cycle_plus": 0, "x_frontier": 0, "s_over_c": 1.0, "depth": 1},
        "dhol_mu": 0.0,
        "prev_xf": 0,
        "prev_ucr": 0.0
    }

def save_state(d):
    STATE.write_text(json.dumps(d, indent=2), encoding="utf-8")

def load_cfg():
    if CFG.is_file():
        try:
            c = json.loads(CFG.read_text("utf-8"))
            return {**DEFAULT_CFG, **c}
        except:
            pass
    return DEFAULT_CFG

def verdict(digest, telem, cfg):
    if digest["cycle_plus"] > 0:
        return "red"
    red = (
        (telem["kappa_eff"] >= cfg["tau_k"] and digest["ucr"] >= cfg["ucr_min"] and telem["evidence_strength"] < cfg["es_min"])
        or (telem["delta_hol"] >= cfg["tau_hol"] and telem["del_suspect"])
    )
    if red:
        return "red"
    amber = any([
        telem["kappa_eff"] >= cfg["tau_k"],
        telem["delta_hol"] >= cfg["tau_hol"],
        digest["ucr"] >= cfg["ucr_min"],
    ])
    return "amber" if amber else "green"

def main():
    # 1) text
    src = Path("docs/input.txt")
    text = src.read_text("utf-8") if src.is_file() else (
        "Claim: Our adapter builds a small reasoning graph. "
        "Because we link evidence with simple markers, we can estimate structure quality. "
        "Therefore the guard can compare density against structure without chain-of-thought."
    )

    # 2) graph + digest
    nodes, edges = build_graph(text)
    N, E = len(nodes), len(edges)
    S = sum(1 for e in edges if e["rel"] == "supports")
    X = sum(1 for e in edges if e["rel"] == "contradicts")
    C = count_cycles(nodes, edges)
    D = max_depth(nodes, edges)
    digest = {"b0": 1, "cycle_plus": C, "x_frontier": X, "s_over_c": (S / max(1, X)), "depth": D}

    # 3) second-order features
    ES = evidence_strength(text)
    UCR = unsupported_claim_ratio(text)
    phi_t = compute_phi_topo(N, E, D, C, X, S)
    phi_s = phi_sem_v2(text)
    Hs = structural_entropy(text)
    rate = rhetorical_rate(text)
    L = len(TOK.findall(text or ""))

    # 4) stateful bits
    st = load_state()
    mu, var = st["mu_len"], st["var_len"]
    prev_digest = st["prev_digest"]

    # Δ_hol on vector incl. ucr (robust on first run)
    vec_keys = ("b0", "cycle_plus", "x_frontier", "s_over_c", "depth")
    if isinstance(prev_digest, dict):
        p = [prev_digest.get(k, 0) for k in vec_keys]
        p.append(st.get("prev_ucr", 0.0))
        q = [digest.get(k, 0) for k in vec_keys]
        q.append(UCR)
        dhol_raw = jsd_vec(p, q, 1e-9)
    else:
        dhol_raw = 0.0
    dhol_mu = 0.75 * st.get("dhol_mu", 0.0) + 0.25 * dhol_raw

    # deletion suspect: contradictions dropped and no resolution language present
    del_sus = (st.get("prev_xf", 0) > X) and (not RESOLVE.search(text))

    # 5) kappa
    kappa = compute_kappa(L, mu, var, Hs, rate, UCR, phi_t, phi_s)

    # 6) telemetry + digest finalize
    digest["ucr"] = UCR
    telem = {
        "phi_topo": phi_t,
        "phi_sem": phi_s,
        "kappa_eff": kappa,
        "delta_hol": dhol_mu,
        "evidence_strength": ES,
        "del_suspect": bool(del_sus),
        "cost_tokens": L,
    }

    # 7) verdict + receipt update
    cfg = load_cfg()
    status = verdict(digest, telem, cfg)

    rec = {}
    if REC.is_file():
        try:
            rec = json.loads(REC.read_text("utf-8"))
        except:
            rec = {}
    rec["receipt_version"] = rec.get("receipt_version") or "olr/1.4L"
    rec.setdefault("claim", "Lean guard receipt.")
    rec["openline_frame"] = {"digest": digest, "telem": telem, "t_logical": int(time.time())}
    rec["status"] = status
    REC.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[ok] 1.4L → status={status}  kappa={kappa:.3f}  dhol={dhol_mu:.3f}  ucr={UCR:.2f}  es={ES:.2f}")

    # 8) persist state (EWMA)
    alpha = 0.2
    mu2 = (1 - alpha) * mu + alpha * L
    var2 = (1 - alpha) * var + alpha * ((L - mu2) ** 2)
    save_state({"mu_len": mu2, "var_len": var2, "prev_digest": digest, "dhol_mu": dhol_mu, "prev_xf": X, "prev_ucr": UCR})

if __name__ == "__main__":
    Path("docs").mkdir(parents=True, exist_ok=True)
    main()
