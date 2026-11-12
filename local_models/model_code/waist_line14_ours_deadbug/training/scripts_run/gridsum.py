import json, glob
rows = []
for f in sorted(glob.glob('results/finetune_grid/grid_*.json')):
    d = json.load(open(f)); a = d['args']; r = d['res']
    rows.append((a['out'].replace('grid_', ''), a['lr'], a['epochs'], a['replay_ratio'], r['poser']['dip_val'], r['poser']['amass_val'], r['joints']['dip_val'], r['joints']['amass_val']))
ref = [r for r in rows if r[0] == 'ref'][0]
print('config                         lr     ep  repl | Poser DIP-val  AMASS-val(x ref) | Joints DIP-val  AMASS-val(x ref)')
for r in sorted(rows, key=lambda r: r[4]):
    print(f'{r[0]:28s} {r[1]:7.0e} {r[2]:3d} {r[3]:4d} |   {r[4]:.5f}   {r[5]:.5f} ({r[5]/ref[5]:.2f}x)  |   {r[6]:.5f}   {r[7]:.5f} ({r[7]/ref[7]:.2f}x)')
