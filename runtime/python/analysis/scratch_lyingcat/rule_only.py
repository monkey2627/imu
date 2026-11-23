"""Re-run only LyingCatContact on captured (anchored network output, calibrated sensors) of a capture npz.
Used for quick variant experiments; verified equal to the live chain on the fix2 capture first."""
import importlib, json, numpy as np


def run(stem, tag="fix2", module="production.lyingcat_contact", patch=None, dt=2 / 60):
    mod = importlib.import_module(module)
    if patch:
        mod = importlib.reload(mod)
        for k, v in patch.items(): setattr(mod, k, v)
    d = np.load(f"analysis/scratch_lyingcat/cap_{stem}_{tag}.npz")
    rule = mod.LyingCatContact()
    labels = list(d["labels"]); out, diags = [], []
    resets = _reset_times(stem)
    t = d["t"]; k = 0
    for i in range(len(t)):
        while k < len(resets) and resets[k] <= t[i]:
            rule.reset("replay"); k += 1
        m = d["measured"][i]
        local, diag = rule.apply(d["anchored"][i].astype(np.float64), dt,
                                 measured=None if np.isnan(m).any() else m, labels=labels)
        out.append(local); diags.append(diag)
    return t, np.array(out), diags


def _reset_times(stem):
    """Times at which live_replay(simulate) resets the rules (rebases / calibration starts / lifecycle resets)."""
    import production.live_replay as lr
    ts = []
    for line in open(f"recordings/{stem}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "event" and (r.get("name") in lr.LEG_RESET_EVENTS or r.get("name") == "modelContextRebase"
                                         or (r.get("name") == "actionLifecycle" and r.get("modelSessionReset"))):
            ts.append(r["t"])
    return ts
