"""Standard-protocol fine-tuning on the DIP-IMU TRAIN subjects (s_01 ... s_07; s_08 is the validation subject, s_09 / s_10 are the
test split and are never touched).  Same as every compared method: pre-train on synthetic AMASS, fine-tune Poser and Joints on the
real IMU recordings of the DIP training subjects.  Sensors = DIP [7,8,11,12,2] (lower arms, lower legs, pelvis), 30 fps.

  python -m mobileposer.finetune_dip --poser pnp5_base_poser --joints pnp5_base_joints --out pnp5_ft --lr 5e-5 --epochs 20
Writes checkpoints/<out>_poser|joints/1/<module>/epoch=*-validation_step_loss=*.ckpt (best validation = best_ckpt()).
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.data import PoseDataset, pad_seq, lowpass_full_sequence
from mobileposer.evaluate_canon_models import best_ckpt
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.models.joints import Joints
from mobileposer.models.poser import Poser

TRAIN = [f's_{i:02d}' for i in range(1, 8)]
VAL = ['s_08']


def build(body, subjects, window=int(__import__('os').environ.get('MOBILEPOSER_WINDOW', 125))):
    data = {k: [] for k in ['imu_inputs', 'pose_outputs', 'joint_outputs', 'slow_joint_outputs', 'tran_outputs', 'vel_outputs', 'foot_outputs']}
    for s in load_dip(body, subjects=subjects):
        T = len(s['pose'])
        G, J = body.forward_kinematics(s['pose'], torch.zeros(10), tran=None, calc_mesh=False)
        slow_J = lowpass_full_sequence(J.reshape(T, 24, 3), int(__import__('os').environ.get('MOBILEPOSER_FPS', 30)), float(__import__('os').environ.get('MOBILEPOSER_SLOW_CUTOFF', 0.2)), int(__import__('os').environ.get('MOBILEPOSER_SLOW_KERNEL', 61)))
        ns = s['acc'].shape[1]
        imu = torch.cat([(s["acc"] / 30.0).reshape(T, 3 * ns), s["ori"].reshape(T, 9 * ns)], dim=1)
        if __import__("os").environ.get("MOBILEPOSER_DOMAIN") == "1":      # annotation-convention token: DIP-style labels
            imu = torch.cat([imu, torch.tensor([0.0, 1.0]).expand(T, -1)], dim=1)
        for key, val in (('imu_inputs', imu), ('pose_outputs', G.reshape(T, 24, 3, 3)), ('joint_outputs', J.reshape(T, 24, 3)), ('slow_joint_outputs', slow_J),
                         ('tran_outputs', torch.zeros(T, 3))):
            data[key].extend(torch.split(val, window))
    ds = PoseDataset.__new__(PoseDataset)
    ds.evaluate, ds.finetune, ds.data = None, 'dip', data
    return ds


class Replay(torch.utils.data.Dataset):
    """DIP windows + randomly drawn AMASS windows (replay): fine-tune on real IMU data without forgetting the synthetic distribution"""
    def __init__(self, dip, amass, ratio, pool=None):
        self.dip, self.amass, self.n = dip, amass, len(dip) * (1 + ratio)
        self.pool = pool if pool is not None else list(range(len(amass)))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < len(self.dip):
            return self.dip[i]
        item = list(self.amass[self.pool[int(torch.randint(len(self.pool), (1,)))]][:4])
        if len(self.amass.data.get('slow_joint_outputs', [])):
            item.append(self.amass.data['slow_joint_outputs'][self.pool[int(torch.randint(len(self.pool), (1,)))]])
        if __import__("os").environ.get("MOBILEPOSER_DOMAIN") == "1":      # AMASS-style (synthetic) token
            item[0] = torch.cat([item[0], torch.tensor([1.0, 0.0]).expand(len(item[0]), -1)], dim=1)
        return tuple(item)


def mean_loss(model, ds, bs=32):
    dev = torch.device("cuda:0")
    model.to(dev).eval()                                   # Lightning leaves the model on the CPU after fit
    ls = []
    with torch.no_grad():
        for (imu, lens), (out, ol) in DataLoader(ds, batch_size=bs, collate_fn=pad_seq):
            ls.append(model.shared_step(((imu.to(dev), lens), ({k: v.to(dev) for k, v in out.items()}, ol))).item())
    return float(sum(ls) / len(ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass_val", type=int, default=0, help="hold out N AMASS windows (never replayed) to measure how much synthetic-data performance is retained")
    ap.add_argument('--poser', required=True); ap.add_argument('--joints', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--init_root', default='checkpoints', help='root containing the pretrained poser/joints checkpoints')
    ap.add_argument('--output_root', default='checkpoints', help='isolated root for all fine-tuning outputs')
    ap.add_argument('--train_all', action='store_true', help='fine-tune on all eight DIP training subjects s_01..s_08 (the protocol of the compared methods); s_08 is then only logged')
    ap.add_argument('--replay_cache', default=None, help='cached AMASS training windows (MOBILEPOSER_DATA_CACHE file)'); ap.add_argument('--replay_ratio', type=int, default=0, help='AMASS windows per DIP window')
    a = ap.parse_args()
    L.seed_everything(a.seed)
    body = ParametricModel(paths.smpl_file)
    train, val = build(body, TRAIN + (VAL if a.train_all else [])), build(body, VAL)
    amv = None
    if a.replay_cache and (a.replay_ratio > 0 or a.amass_val > 0):
        am = PoseDataset.__new__(PoseDataset)
        am.evaluate, am.finetune, am.data = None, None, torch.load(a.replay_cache)
        perm = torch.randperm(len(am), generator=torch.Generator().manual_seed(0)).tolist()
        vidx, pool = perm[:a.amass_val], perm[a.amass_val:]
        if a.amass_val > 0:
            amv = PoseDataset.__new__(PoseDataset)
            amv.evaluate, amv.finetune = None, None
            amv.data = {k: [v[i] for i in vidx] for k, v in am.data.items()}
        if a.replay_ratio > 0:
            train = Replay(train, am, a.replay_ratio, pool)
    res = {}
    print("windows: train", len(train), "val", len(val), flush=True)
    for name, cls, src in (('poser', Poser, a.poser), ('joints', Joints, a.joints)):
        if __import__("os").environ.get("MOBILEPOSER_DOMAIN") == "1":
            model = cls()                                                   # one extra input pair: pad the pre-trained input weights with zeros
            sd = torch.load(best_ckpt(str(Path(a.init_root) / src)), map_location="cpu")["state_dict"]
            new = model.state_dict()
            for k, v in sd.items():
                if new[k].shape != v.shape:
                    pad = torch.zeros_like(new[k]); pad[:, :v.shape[1]] = v; sd[k] = pad
            model.load_state_dict(sd)
        else:
            model = cls.load_from_checkpoint(best_ckpt(str(Path(a.init_root) / src)), map_location="cpu")
        model.hypers = SimpleNamespace(batch_size=a.batch, lr=a.lr)
        d = Path(a.output_root) / f'{a.out}_{name}/1/{name}'
        d.mkdir(parents=True, exist_ok=True)
        ck = ModelCheckpoint(monitor='validation_step_loss', save_top_k=3, mode='min', dirpath=str(d), save_weights_only=True,
                             filename='{epoch}-{validation_step_loss:.4f}')
        trainer = L.Trainer(max_epochs=a.epochs, min_epochs=a.epochs, devices=[0], accelerator='gpu', callbacks=[ck],
                            logger=False, enable_progress_bar=False, deterministic=True)
        trainer.fit(model, DataLoader(train, batch_size=a.batch, collate_fn=pad_seq, shuffle=True),
                    DataLoader(val, batch_size=a.batch, collate_fn=pad_seq))
        print(name, "best", ck.best_model_path, ck.best_model_score, flush=True)
        res[name] = {"dip_val": mean_loss(model, val), "amass_val": mean_loss(model, amv) if amv is not None else None}
        print(name, "final-model losses", res[name], flush=True)
    import json
    Path("results/finetune_grid").mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a), "res": res}, open(f"results/finetune_grid/{a.out}.json", "w"))


if __name__ == '__main__':
    main()
