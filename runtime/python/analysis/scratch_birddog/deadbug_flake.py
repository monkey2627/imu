import sys, numpy as np
import production.birddog_contact as bc
import production.live_replay as lr
rec = sys.argv[1]
r = lr.replay_live(rec); print("with observe:", r.summary())
bad = np.where(r.rules_known & (r.rules_error_deg > 0.5))[0]; print("bad idx/times", bad, r.times[bad])
bc.BirdDogContact.observe = lambda self, labels, acc: None
r2 = lr.replay_live(rec); print("observe disabled:", r2.summary())
