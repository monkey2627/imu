"""Per cat/cow cycle (adjacent cat & cow phases): waist pitch diff, output spine total pitch diff
(spine3 - pelvis elevation, i.e. how much the trunk bends), pelvis pitch diff, arch diff (cm), for raw/prod/fix."""
import numpy as np
from analysis.scratch_lyingcat.cycles import PHASES, layer_signals
import sys
for stem, ph in [(k, PHASES[k]) for k in (sys.argv[1:] or PHASES)]:
    S = {tag: layer_signals(stem, tag) for tag in ("prod", "fix3")}
    def m(tag, k, a, b):
        tt, v = S[tag][k]; return v[(tt >= a) & (tt < b)].mean()
    print(f"\n{stem}: cycle = cat phase minus the cow phase next to it (cat-cow)")
    print("  cycle   cat t / cow t  | waist | net trunk net pelvis net arch | prod trunk prod arch | fix trunk fix arch | pelvis(out)")
    for i in range(0, len(ph) - 1):
        c, w = (ph[i], ph[i + 1]) if ph[i][0] == "cat" else (ph[i + 1], ph[i])
        if c[0] != "cat" or w[0] != "cow": continue
        f = lambda tag, k: m(tag, k, *c[1:]) - m(tag, k, *w[1:])
        print(f"  {i:2d}  {c[1]:5.1f} / {w[1]:5.1f}  | {f('fix3','waist calib'):+5.1f} | {f('prod','net trunk'):+7.1f} {f('prod','net pelvis'):+8.1f} {f('prod','net arch cm'):+7.1f}"
              f" | {f('prod','out trunk'):+9.1f} {f('prod','out arch cm'):+8.1f} | {f('fix3','out trunk'):+8.1f} {f('fix3','out arch cm'):+7.1f} | {f('fix3','out pelvis'):+6.1f}")
    for tag in ("prod", "fix3"):
        a = [m(tag, "out arch cm", a, b) for n, a, b in ph]
        cats = [x for x, p in zip(a, ph) if p[0] == "cat"]; cows = [x for x, p in zip(a, ph) if p[0] == "cow"]
        print(f"  [{tag}] arch cm: cat phases {np.round(cats,1)}  cow phases {np.round(cows,1)}  neutral hold 9-10s {m(tag,'out arch cm',9,10):+.1f}")
