import re, numpy as np
ds = ['dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib']
def parse(f):
    out = {}
    try:
        for line in open(f, errors='ignore'):
            m = re.match(r'(dipimu|totalcapture_dipcalib|totalcapture_officalib) SIP/Ang/Pos/Mesh \[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]', line)
            if m: out[m.group(1)] = [float(x) for x in m.groups()[1:]]
    except FileNotFoundError: pass
    return out
def files(kind, tag):
    # kind: base / sc ; tag: '' , _ftr3 , _ftr1
    if kind == 'base': return [f'std_pnp5_60a_s{s}{tag}.log' for s in (42, 7, 3)]
    return [f'std_pnp5_60a_sc{tag}.log' if s == 42 else f'std_pnp5_60a_s{s}sc{tag}.log' for s in (42, 7, 3)]
for tag, name in [('', 'pre-trained'), ('_ftr3', 'fine-tuned + replay x3'), ('_ftr1', 'fine-tuned + replay x1')]:
    print('==', name)
    for kind in ('base', 'sc'):
        rows = [parse(f) for f in files(kind, tag)]
        for d in ds:
            v = np.array([r[d] for r in rows if d in r])
            if len(v): print(f'  {kind:4s} {d:24s} ' + ' / '.join(f'{x:5.2f}' for x in v.mean(0)) + '   std ' + ' / '.join(f'{x:4.2f}' for x in v.std(0)) + f'  n={len(v)}')
