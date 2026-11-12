"""Line 14 environment-B measurement: B0/B1/Oracle with the 12-d R1 representation (validation-only, shape-aware labels)."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from mobileposer.line9.data import (load_identity_records, PairedWindowDataset,
    collate_paired_windows, SubsetBatchSampler)
from mobileposer.line9.store import SequenceStore
from mobileposer.line9.models import build_fair_models, fit_geometry_stats, parameter_count_report
from mobileposer.line9.evaluation import PositionEvaluationAccumulator
from mobileposer.line14.r1 import R1Table
R1 = R1Table()

def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def geometry_rows(dataset, records):
    rows=[]; seen=set()
    for rec in records:
        if rec.identity not in seen:
            beta, _lengths=dataset._get_geometry(rec)
            rows.append(R1.get(beta))
            seen.add(rec.identity)
    return torch.stack(rows)

def loader(dataset,batch,seed,shuffle):
    sampler=SubsetBatchSampler(dataset,batch,shuffle=shuffle,drop_last=False,seed=seed)
    return DataLoader(dataset,batch_sampler=sampler,collate_fn=collate_paired_windows,num_workers=0,pin_memory=True),sampler

def evaluate(model, dl, device, seed, bootstrap=10000):
    model.eval(); acc=PositionEvaluationAccumulator()
    with torch.no_grad():
        for b in dl:
            x=b['target_imu'].to(device); geom=R1.batch(b['beta']).to(device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                p=model(x,b['target_lengths'].tolist(),geom)
            acc.update(p.cpu().reshape(*p.shape[:2],24,3),b['target_joints'],b['target_mask'],b['identity'])
    return acc.compute(n_bootstrap=bootstrap,seed=20261013+seed)

def main():
    a=argparse.ArgumentParser(); a.add_argument('--mode',choices=['b0','b1','oracle'],required=True); a.add_argument('--seed',type=int,required=True); a.add_argument('--gpu',type=int,default=0)
    a.add_argument('--epochs',type=int,default=60); a.add_argument('--batch',type=int,default=128); a.add_argument('--lr',type=float,default=1e-3); a.add_argument('--weight-decay',type=float,default=.01); a.add_argument('--window',type=int,default=125)
    a.add_argument('--data-root',type=Path,default=Path('data/pnp5_60a/pnp5')); a.add_argument('--audit',type=Path,default=Path('results/line9/data_audit.json')); a.add_argument('--out-root',type=Path,default=Path('results/line14/envB')); a.add_argument('--checkpoint-root',type=Path,default=Path('checkpoints/line14'))
    a.add_argument('--max-identities',type=int); a.add_argument('--max-sequences',type=int); a.add_argument('--max-windows',type=int); a.add_argument('--store',type=Path,default=Path('data/line9_store_shape')); a.add_argument('--smoke',action='store_true'); a=a.parse_args()
    if a.smoke: a.epochs=1; a.max_identities=a.max_identities or 2; a.max_sequences=a.max_sequences or 2; a.max_windows=a.max_windows or 8; a.batch=min(a.batch,4)
    seed_all(a.seed); device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    records=load_identity_records(a.data_root,a.audit,('train','validation'),max_identities_per_fold=a.max_identities,max_sequences_per_identity=a.max_sequences)
    store=SequenceStore(a.store) if (a.store/'index.json').is_file() else None
    print('sequence store:',a.store if store else None,flush=True)
    train=PairedWindowDataset(records['train'],a.window,max_windows=a.max_windows,file_cache_size=2,sequence_cache_size=64,store=store)
    val=PairedWindowDataset(records['validation'],a.window,max_windows=a.max_windows,file_cache_size=2,sequence_cache_size=64,store=store)
    stats=fit_geometry_stats(geometry_rows(train,records['train']))
    models=build_fair_models(stats); counts=parameter_count_report(models); model=models[a.mode].to(device); del models
    td,ts=loader(train,a.batch,a.seed,True); vd,vs=loader(val,a.batch,a.seed,False)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    out=a.out_root/f'{a.mode}_s{a.seed}.json'; ck=a.checkpoint_root/f'{a.mode}_s{a.seed}.pt'; out.parent.mkdir(parents=True,exist_ok=True); ck.parent.mkdir(parents=True,exist_ok=True)
    best=float('inf'); history=[]; started=time.time()
    for epoch in range(a.epochs):
        ts.set_epoch(epoch); model.train(); total=n=0
        for b in td:
            x=b['target_imu'].to(device); y=b['target_joints'].to(device).flatten(2); mask=b['target_mask'].to(device); geom=R1.batch(b['beta']).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                p=model(x,b['target_lengths'].tolist(),geom); mse=((p-y).square()*mask[:,:,None]).sum()/(mask.sum()*72)
                acc=p[:,2:]+p[:,:-2]-2*p[:,1:-1]; tm=mask[:,2:]&mask[:,:-2]&mask[:,1:-1]; temporal=(acc.abs()*tm[:,:,None]).sum()/tm.sum().clamp_min(1)
                loss=mse+1e-5*temporal
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=float(loss)*int(mask.sum()); n+=int(mask.sum())
        # Cheap validation loss for model selection; full bootstrap only once.
        model.eval(); ve=vn=0
        with torch.no_grad():
            for b in vd:
                x=b['target_imu'].to(device); y=b['target_joints'].to(device).flatten(2); m=b['target_mask'].to(device); g=R1.batch(b['beta']).to(device)
                p=model(x,b['target_lengths'].tolist(),g); e=(p.reshape(*p.shape[:2],24,3)-y.reshape(*y.shape[:2],24,3)).norm(dim=-1)
                ve+=float((e*m[:,:,None]).sum()); vn+=int(m.sum())*24
        score=100*ve/vn; history.append({'epoch':epoch+1,'train_loss':total/max(n,1),'validation_all_cm_pooled':score})
        if score<best: best=score; torch.save({'model':model.state_dict(),'epoch':epoch+1,'score':score,'geometry_mean':stats.mean,'geometry_std':stats.std},ck)
        print(json.dumps({'mode':a.mode,'seed':a.seed,**history[-1]}),flush=True)
    state=torch.load(ck,map_location=device,weights_only=False); model.load_state_dict(state['model']); metrics=evaluate(model,vd,device,a.seed,10 if a.smoke else 10000)
    result={'status':'smoke_only' if a.smoke else 'formal_validation_only','mode':a.mode,'seed':a.seed,'protocol':{'epochs':a.epochs,'batch':a.batch,'window':a.window,'optimizer':'AdamW','lr':a.lr,'weight_decay':a.weight_decay,'train_identities':len({r.identity for r in records['train']}),'validation_identities':len({r.identity for r in records['validation']}),'train_sequences':len(records['train']),'validation_sequences':len(records['validation']),'target_window_digest_train':train.target_window_digest(),'target_window_digest_validation':val.target_window_digest(),'test_accessed':False},'parameter_counts':counts,'best_epoch':state['epoch'],'history':history,'metrics':metrics,'elapsed_seconds':time.time()-started}
    tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)); os.replace(tmp,out)
if __name__=='__main__': main()
