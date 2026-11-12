"""Tiny GPU job queue for line 14.

usage: python scripts_run/line14_sched.py JOBS.txt --gpus 0,1,2,3,4,6,7 [--log-dir logs/line14/E2]
JOBS.txt: one shell command per line; the literal token {GPU} is replaced by the assigned GPU index.
Lines starting with '#' are ignored. A job's stdout/stderr goes to <log-dir>/<n>.log.
Does not kill or touch anything outside its own children.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('jobs'); p.add_argument('--gpus', default='0'); p.add_argument('--log-dir', default='logs/line14/sched')
a = p.parse_args()
jobs = [l.strip() for l in open(a.jobs, encoding='utf8') if l.strip() and not l.startswith('#')]
free = [int(x) for x in a.gpus.split(',')]
Path(a.log_dir).mkdir(parents=True, exist_ok=True)
running = {}  # proc -> (gpu, idx)
done = failed = 0
nxt = 0
env = dict(os.environ)
while nxt < len(jobs) or running:
    while nxt < len(jobs) and free:
        g = free.pop(0); cmd = jobs[nxt].replace('{GPU}', str(g))
        log = open(Path(a.log_dir) / f'{nxt:03d}.log', 'w')
        pr = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        running[pr] = (g, nxt); print(f'[start] job {nxt} on gpu {g}: {cmd[:110]}', flush=True); nxt += 1
    time.sleep(5)
    for pr in list(running):
        if pr.poll() is not None:
            g, i = running.pop(pr); free.append(g)
            if pr.returncode == 0: done += 1
            else: failed += 1
            print(f'[end] job {i} rc={pr.returncode} (done {done}, failed {failed})', flush=True)
print(f'ALL DONE: {done} ok, {failed} failed', flush=True)
sys.exit(1 if failed else 0)
