import re, glob, numpy as np
ds = ['dipimu', 'totalcapture_dipcalib', 'totalcapture_officalib']
def parse(f):
    out = {}
    for line in open(f, errors='ignore'):
        m = re.match(r'(dipimu|totalcapture_dipcalib|totalcapture_officalib) SIP/Ang/Pos/Mesh \[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]', line)
        if m: out[m.group(1)] = [float(x) for x in m.groups()[1:]]
    return out
for tag, name in [('', 'pre-trained'), ('_ft', 'DIP fine-tune, no replay'), ('_ftr1', 'DIP fine-tune + replay x1'), ('_ftr3', 'DIP fine-tune + replay x3')]:
    rows = [parse(f'std_pnp5_60a_s{s}{tag}.log') for s in (42, 7, 3)]
    print('==', name, ' (seeds done:', sum(len(r) == 3 for r in rows), ')')
    for d in ds:
        v = np.array([r[d] for r in rows if d in r])
        if len(v): print(f'  {d:24s} mean ' + ' / '.join(f'{x:5.2f}' for x in v.mean(0)) + '   std ' + ' / '.join(f'{x:4.2f}' for x in v.std(0)) + f'   (n={len(v)})')
