"""Lying cat: ① network raw | ② production before the fix (git 48624de) | ③ fix (waist cat/cow relative
to the neutral hold, production/lyingcat_contact.py 2026-11-23).  All three from
production.live_replay.replay_live(simulate=True) (current run.py logic on the recorded input), captured by
analysis/scratch_lyingcat/capture.py (tags raw-from-prod / prod / fix).  Top: side view, bottom: 3/4 view.
Per-frame text: waist pitch, output pelvis / spine1 / spine3 pitch and the mid-back arch (cm above the
pelvis->neck chord, + = cat), plus the phase from the waist's vertical translation.

    python -m analysis.lyingcat_neutral_compare_video [stem ...] [--workers 8]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from analysis.compare_video import render
from analysis.scratch_lyingcat.cycles import PHASES
from analysis.scratch_lyingcat.signal import elev
from analysis.scratch_lyingcat.timeline import metrics
from production.paths import paths

SUBTITLES = {
    "lyingcat_20261123_044239": "044239（waist 戴在腰最细处前侧）：起始中立 0-14s，猫/牛交替约 5 次（后 3 次幅度更大），48s 起身",
    "lyingcat_20261123_044417": "044417（同一佩戴）：12s 重新校准，中立 20-23s，之后猫/牛交替约 4 次，58s 起身",
    "lyingcat_20261120_230158": "230158（旧佩戴）：起始中立 0-10s，之后猫/牛交替约 5 次，52s 起身。waist 俯仰猫-牛差约 18°",
    "lyingcat_20261123_004720": "004720（重新佩戴）：起始中立 0-12s，之后猫/牛交替约 5 次，57s 起身。waist 俯仰猫-牛差仅约 3.5°（垂直位移照常）",
}


def phase_at(stem, t):
    for name, a, b in PHASES.get(stem, []):
        if a <= t < b:
            return "CAT (arch up)" if name == "cat" else "COW (sag)"
    return ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", default=list(SUBTITLES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for stem in args.stems:
        prod = np.load(f"analysis/scratch_lyingcat/cap_{stem}_prod.npz")
        fix = np.load(f"analysis/scratch_lyingcat/cap_{stem}_fix3.npz")
        t = fix["t"]
        assert np.array_equal(t, prod["t"]) and np.array_equal(fix["raw"], prod["raw"])
        waist = elev(fix["measured"][:, 4, :, 1])
        columns = [("raw", "① 网络原始输出", fix["raw"].astype(np.float64)),
                   ("prod", "② 当前生产（修复前 48624de）", prod["final"].astype(np.float64)),
                   ("fix", "③ 修复后：俯仰+侧倾相对中立位", fix["final"].astype(np.float64))]
        text = {}
        for key, _title, local in columns:
            m, _ = metrics(local)
            lines = []
            for i in range(len(t)):
                row = [f"t {t[i]:5.1f}s  waist {waist[i]:+5.1f} deg  {phase_at(stem, t[i])}",
                       f"pelvis {m['pelv'][i]:+5.1f} spine1 {m['sp1'][i]:+5.1f} spine3 {m['sp3'][i]:+5.1f}",
                       f"arch {m['arch9'][i]:+5.1f} cm (+ = cat)"]
                if key == "fix":
                    d = json.loads(fix["diag"][i]) if fix["diag"][i] != "null" else None
                    wn = (d or {}).get("waistNeutral") or {}
                    if wn.get("waistDeltaDeg") is not None:
                        row[-1] += f"  dWaist {wn['waistDeltaDeg']:+5.1f}"
                lines.append(row)
            text[key] = lines
        # Real-time playback: one video frame per 1/30 s of recording time, showing the latest output
        # then.  (Live runs with inferEvery=1 give ~50 outputs/s; one frame per output played them
        # 1.7x slow -- 2026-11-23, 044239/044417.)
        grid = np.arange(t[0], t[-1], 1.0 / 30.0)
        pick = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
        columns = [(key, title, local[pick]) for key, title, local in columns]
        text = {key: [lines[i] for i in pick] for key, lines in text.items()}
        out = paths.recordings / f"lyingcat_neutral_compare_{stem}.mp4"
        render(out, columns, grid, text, SUBTITLES.get(stem, stem), workers=args.workers)
        print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
