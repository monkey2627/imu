import sys, numpy as np
import production.birddog_contact as bc
import production.live_replay as lr
side = sys.argv[2]; t0, t1 = float(sys.argv[3]), float(sys.argv[4])
orig_obs = bc.BirdDogContact.observe
seglog = []
def obs(self, labels, acc):
    before = len(self.events[side])
    orig_obs(self, labels, acc)
    if len(self.events[side]) > before and self is chain_ref[0]:
        seglog.append((now[0], self.events[side][-1]))
chain_ref = [None]; now = [0.0]
orig_apply = bc.BirdDogContact.apply
rows = []
def app(self, local, dt):
    if chain_ref[0] is None: chain_ref[0] = self
    out = orig_apply(self, local, dt)
    if self is chain_ref[0]:
        d = out[1]; rows.append((d["sensorElevationDeg"].get(f"{side}RiseDeg"), d["lifted"][side], d["shankHeightM"][side], d["weights"][side]))
    return out
bc.BirdDogContact.observe = obs; bc.BirdDogContact.apply = app
# track time via tick parsing: patch prepare_frame not easy; use rows order + r.times
r = lr.replay_live(sys.argv[1], simulate=True)
n = len(r.times)
rows = rows[:n] if len(rows) >= n else rows
for i, t in enumerate(r.times):
    if t0 <= t <= t1 and i % 6 == 0 and i < len(rows):
        print(f"{t:6.2f} rise {rows[i][0]} up {rows[i][1]} h {rows[i][2]} w {rows[i][3]}")
