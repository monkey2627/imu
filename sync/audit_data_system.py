"""Read-only, no-follow source/link inventory for the server data system."""
import os
import json
from pathlib import Path

BASE = Path('/home/duanyuhan/dyh/motion')
ROOT = BASE / 'data_system'
PAIRS = [('flowhmr', 'core/flowhmr'), ('duomo', 'models/duomo'),
         ('gvhmr', 'models/gvhmr'), ('lines/baseline_20261010', 'integrations/mobileposer')]
for parent in [Path('/'), *reversed(ROOT.parents), ROOT]:
    f = parent / 'AGENTS.md'
    if f.is_file():
        print('INSTRUCTIONS', str(f), f.read_text())
print('ROOT_LAYOUT')
for d, dirs, files in os.walk(ROOT, followlinks=False):
    if len(Path(d).relative_to(ROOT).parts) >= 2:
        dirs[:] = []
    for name in sorted(dirs + files):
        p = Path(d) / name
        print('ENTRY', 'link' if p.is_symlink() else 'dir' if p.is_dir() else 'file', str(p), os.readlink(p) if p.is_symlink() else '')
print('CANONICAL_LINK_AUDIT')
for d, dirs, files in os.walk(ROOT, followlinks=False):
    for name in sorted(dirs + files):
        p = Path(d) / name
        if p.is_symlink():
            print('LINK', str(p.relative_to(ROOT)), os.readlink(p), 'OK' if p.exists() else 'DANGLING')
        elif name == 'AGENTS.md':
            print('INSTRUCTIONS', str(p), p.read_text())
RESOURCE = {
    'flowhmr': {'assets', 'checkpoints', 'ckpts', 'output'},
    'duomo': {'data', 'results', '.incoming_assets_20261107'},
    'gvhmr': {'inputs', 'outputs', '.venv'},
    'lines/baseline_20261010': {'data', 'checkpoints', 'logs', 'results'},
}
for old, new in PAIRS:
    src, dst = BASE / old, ROOT / new
    print('CANONICAL', str(dst), 'link=' + str(dst.is_symlink()), 'resolved=' + str(dst.resolve()))
    counts = {'links': 0, 'regular': 0}
    for d, dirs, files in os.walk(src, followlinks=False):
        rel = Path(d).relative_to(src)
        for name in sorted(dirs + files):
            p = Path(d) / name
            r = p.relative_to(src)
            if name in {'.git', '__pycache__', '.pytest_cache'} or name.endswith('.pyc'):
                if name in dirs:
                    dirs.remove(name)
                continue
            if (len(r.parts) == 1 and name in RESOURCE[old]) or (str(rel) == 'third_party/PNP' and name in {'data', 'models', 'drift_results'}):
                print('RESOURCE', str(p), 'link=' + str(p.is_symlink()), 'exists=' + str(p.exists()), 'target=' + (os.readlink(p) if p.is_symlink() else ''))
                if name in dirs:
                    dirs.remove(name)
                continue
            target = dst / r
            if p.is_symlink():
                counts['links'] += 1
                status = 'DANGLING' if not p.exists() else 'CANONICAL' if p.resolve() == target.resolve() else 'OTHER_TARGET'
                print('LEGACY_LINK', str(p), os.readlink(p), status)
            elif p.is_file():
                counts['regular'] += 1
                print('LEGACY_REGULAR', str(p), 'bytes=' + str(p.stat().st_size), 'canonical_exists=' + str(target.exists()))
    print('SUMMARY', old, json.dumps(counts))
pnp = ROOT / 'integrations/mobileposer/third_party/PNP'
for p in sorted(pnp.iterdir()):
    print('PNP_CANONICAL_ENTRY', p.name, 'link' if p.is_symlink() else 'dir' if p.is_dir() else 'file', os.readlink(p) if p.is_symlink() else '')
