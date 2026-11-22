import sys, numpy as np
import production.frame_chain as fc
import production.live_replay as lr
inst = []
orig_init = fc.LegChain.__init__
def init(self, *a, **k):
    orig_init(self, *a, **k); inst.append(self)
fc.LegChain.__init__ = init
orig_obs = fc.LegChain.observe_acceleration
def obs(self, labels, samples, keys, fallback):
    if len(inst) >= 2 and self is inst[1]:     # live_replay's rules chain: behave like HEAD (no call)
        return None
    return orig_obs(self, labels, samples, keys, fallback)
fc.LegChain.observe_acceleration = obs
r = lr.replay_live(sys.argv[1]); print("rules chain not fed (HEAD behaviour):", r.summary())
