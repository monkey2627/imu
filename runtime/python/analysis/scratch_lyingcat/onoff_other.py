"""Other actions: production lyingcat_contact (fix) vs the pre-fix module (v0) must give bit-identical output."""
import sys, numpy as np, torch
import production.frame_chain as fc
import production.live_replay as lr
from analysis.scratch_lyingcat import v0_prod_48624de as v0
torch.set_grad_enabled(False)
NEW = fc.LyingCatContact
for rec in sys.argv[1:]:
    out = {}
    for mode, cls in (("fix", NEW), ("v0", v0.LyingCatContact)):
        fc.LyingCatContact = cls
        out[mode] = lr.replay_live(rec, simulate=True).final_local.copy()
    fc.LyingCatContact = NEW
    print(f"RESULT {rec}: bit-identical={np.array_equal(out['fix'], out['v0'])} maxabs={np.abs(out['fix'] - out['v0']).max():.3g}", flush=True)
