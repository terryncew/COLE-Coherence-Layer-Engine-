from __future__ import annotations
import json, math, os, time, hashlib
from pathlib import Path

REC = Path("docs/receipt.latest.json")
CFG = Path("docs/guard.14l.json")
ST  = Path(".state/hazard_v04.json")
LEDGER = Path("docs/ledger.jsonl")
POLICY = Path("docs/policy.json")

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def zscore(x, mu, var):
    sd = max(1e-9, var) ** 0.5
    return 0.0 if sd == 0 else (x - mu)/sd

def ew(mu, var, x, alpha=0.2):
    mu2 = (1-alpha)*mu + alpha*x
    var2 = (1-alpha)*var + alpha*(x-mu2)**2
    return mu2, var2

def delta_sigma_vec(dhol_dict, w):
    keys = ("prompt","tool","retrieval","model","cache")
    return sum(w.get(k,0.0) * (max(0.0, dhol_dict.get(k,0.0))**2) for k in keys) ** 0.5

def p_review(k_z, d_sig, eps_z, needle_fail, k_spike, lam=None, base_logit=0.0):
    lam = lam or {"base":0.05,"k":0.35,"d":0.35,"e":0.20,"n":0.30,"s":0.25}
    k_z = max(0.0, k_z); d_sig=max(0.0, d_sig); eps_z=max(0.0, eps_z)
    needle_fail = 1 if needle_fail else 0; k_spike = 1 if k_spike else 0
    h = lam["base"] + lam["k"]*k_z + lam["d"]*d_sig + lam["e"]*eps_z + lam["n"]*needle_fail + lam["s"]*k_spike
    x = h + base_logit
    return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))

def badge_from(p, k_window, needle_fail, k_spike, bands):
    if k_window < 3:
        return "white", "white"
    loM, hiM = bands["amber_monitor"]
    loI, hiI = bands["amber_investigate"]
    if p >= hiI or (needle_fail and k_spike):
        return "red", "red"
    if p < bands["green"][1] and not needle_fail and not k_spike:
        return "green", "green"
    if loI <= p < hiI:
        return "amber", "amber-investigate"
    return "amber", "amber-monitor"

def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()

def ulid_like() -> str:
    ts = int(time.time() * 1000)
    rnd = hashlib.sha1(os.urandom(16)).hexdigest()[:12]
    return f"{ts:013d}-{rnd}"

# --- load
rec = json.loads(REC.read_text("utf-8"))
cfg = json.loads(CFG.read_text("utf-8")) if CFG.is_file() else {"es_min":0.25}
policy = {
    "argue_mins": 14, "review_mins": 4, "daily_review_budget": 60,
    "bands": {"green":[0.0,0.20], "amber_monitor":[0.20,0.35], "amber_investigate":[0.35,0.80]}
}
if POLICY.is_file():
    try:
        p = json.loads(POLICY.read_text("utf-8"))
        for k in ("argue_mins","review_mins","daily_review_budget"):
            if k in p: policy[k] = p[k]
        if "bands" in p: policy["bands"].update(p["bands"])
    except: pass

of = rec.get("openline_frame",{})
dig = of.get("digest",{})
tel = of.get("telem",{})

kappa = float(tel.get("kappa_eff", 0.0))
dhol_scalar = float(tel.get("delta_hol", 0.0))
ucr   = float(dig.get("ucr", 0.0))
es    = float(tel.get("evidence_strength", 0.0))

# drift vector (fallback to scalar)
dhol_vec = {"prompt":0.0,"tool":0.0,"retrieval":0.0,"model":0.0,"cache":dhol_scalar}
weights  = {"prompt":0.10,"tool":0.15,"retrieval":0.45,"model":0.10,"cache":0.20}
d_sig = delta_sigma_vec(dhol_vec, weights)

# epsilon_hat ~ unsupported + weak evidence
eps_hat = ucr + max(0.0, (cfg.get("es_min",0.25) - es))

# --- state
ST.parent.mkdir(parents=True, exist_ok=True)
state = {"n":0,"k_mu":0.0,"k_var":1.0,"e_mu":0.0,"e_var":1.0,"k_last":0.0,
         "cusum_k":0.0,"cusum_e":0.0}
if ST.is_file():
    try: state.update(json.loads(ST.read_text("utf-8")))
    except: pass

k_z = zscore(kappa, state["k_mu"], state["k_var"])
e_z = zscore(eps_hat, state["e_mu"], state["e_var"])
k_spike = 1 if (kappa - state.get("k_last", 0.0)) > 0.20 else 0

# base-rate prior
hist_fail = hist_total = 0
hist_badge_red = hist_badge_total = 0
hist_dir = Path("docs/history")
if hist_dir.is_dir():
    for pth in sorted(hist_dir.glob("receipt-*.json"))[-200:]:
        try:
            j = json.loads(pth.read_text("utf-8"))
            he = j.get("human_eval",{})
            v = (he.get("verdict") or "").lower()
            if v in ("fail","borderline"): hist_fail += 1
            if he: hist_total += 1
            b = (j.get("badge") or j.get("status") or "").lower()
            if b in ("red","amber-investigate"): hist_badge_red += 1
            if b: hist_badge_total += 1
        except: pass
if hist_total >= 10:
    pi = hist_fail / max(1, hist_total)
elif hist_badge_total >= 10:
    pi = hist_badge_red / max(1, hist_badge_total)
else:
    pi = 0.15
base_logit = math.log(max(1e-6, pi) / max(1e-6, 1.0 - pi))

p = p_review(k_z, d_sig, e_z, needle_fail=0, k_spike=k_spike, base_logit=base_logit)
badge, badge_detail = badge_from(p, min(10, state["n"]+1), needle_fail=False, k_spike=k_spike, bands=policy["bands"])

# cost-aware threshold suggestion
arg_m, rev_m, budget = policy["argue_mins"], policy["review_mins"], policy["daily_review_budget"]
p_list = []
if hist_dir.is_dir():
    for pth in sorted(hist_dir.glob("receipt-*.json"))[-200:]:
        try:
            jj = json.loads(pth.read_text("utf-8"))
            pr = float((jj.get("hazard") or {}).get("p_review", 0.0))
            p_list.append(pr)
        except: pass
if not p_list: p_list = [p]
p_sorted = sorted(p_list, reverse=True)
minutes_per_item = max(0.0, arg_m - rev_m)
if minutes_per_item <= 0:
    tau = policy["bands"]["amber_investigate"][0]
else:
    k_cap = int(budget // max(1, rev_m))
    k_cap = max(1, k_cap)
    pr_cut = p_sorted[min(len(p_sorted)-1, k_cap-1)]
    tau = max(policy["bands"]["amber_monitor"][1], pr_cut)

# write back
rec.setdefault("spec_version", "0.3")
rec.setdefault("compat", {"min":"0.2","max":"0.5"})
rec.setdefault("signature_scope", ["openline_frame","hazard","badge","badge_detail","fingerprints"])

rec["hazard"] = {"p_review": round(p, 3), "k_window": int(min(10, state["n"]+1)), "tau_suggest": round(tau, 3)}
rec["badge"] = badge
rec["badge_detail"] = badge_detail
rec["status"] = badge if badge in ("green","amber","red") else rec.get("status", "white")

evt = of.get("t_logical") or int(time.time())
rec["attest"] = {
    "tee": "none",
    "clock": "git",
    "event_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(evt)),
    "sign_time": _now_iso(),
    "nonce": f"{int(time.time()*1000)}-{hashlib.sha1(os.urandom(16)).hexdigest()[:12]}"
}

REC.write_text(json.dumps(rec, indent=2), encoding="utf-8")

# update state + regime shift
def ph_update(cusum, x, mu, delta=0.1):
    return max(0.0, cusum + (x - mu - delta))

state["cusum_k"] = ph_update(state.get("cusum_k",0.0), kappa, state["k_mu"])
state["cusum_e"] = ph_update(state.get("cusum_e",0.0), eps_hat, state["e_mu"])
shift = (state["cusum_k"] > 1.5) or (state["cusum_e"] > 1.5)
if shift and state["n"] >= 10:
    state["k_mu"], state["k_var"] = kappa, 1.0
    state["e_mu"], state["e_var"] = eps_hat, 1.0
    state["cusum_k"] = state["cusum_e"] = 0.0

state["k_mu"], state["k_var"] = ew(state["k_mu"], state["k_var"], kappa)
state["e_mu"], state["e_var"] = ew(state["e_mu"], state["e_var"], eps_hat)
state["k_last"] = kappa
state["n"] = state.get("n",0) + 1
ST.write_text(json.dumps(state, indent=2), encoding="utf-8")

# append to ledger
try:
    b = REC.read_bytes()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _now_iso(),
        "receipt_sha256": "sha256:" + hashlib.sha256(b).hexdigest(),
        "badge": rec.get("badge"),
        "p_review": rec.get("hazard",{}).get("p_review"),
        "nonce": rec["attest"]["nonce"]
    }
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, separators=(",",":")) + "\n")
except Exception as e:
    print(f"[ledger] skip ({e})")

print(f"[hazard v0.4] p_review={p:.3f} badge={badge_detail} kappa={kappa:.3f} dhol={dhol_scalar:.3f} ucr={ucr:.2f} es={es:.2f} tau={tau:.3f}")
