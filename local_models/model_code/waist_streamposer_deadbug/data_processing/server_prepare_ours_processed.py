#!/usr/bin/env python3
"""Build validated five-sensor inputs and HumanPose SMPL labels for all Ours captures."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--humanpose', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--retarget', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace))
    sys.path.insert(0, str(args.retarget.parent))
    import mobileposer.test_ours_surface as adapter
    adapter.FPS = 60
    prepare = adapter.prepare
    from mobileposer.retarget_ours_humanpose import retarget_cache
    captures = sorted((p for p in args.raw.glob('*/*') if (p / 'manifest.json').exists()),
                      key=lambda p: int(p.parent.name.split('_', 1)[0]))
    if len(captures) != 10:
        raise RuntimeError(f'expected 10 captures, found {len(captures)}')
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = {'kind': 'Ours Unity HumanPose pseudo-label dataset', 'mode': 'calibrated',
               'independent_ground_truth': False, 'captures': []}
    for capture in captures:
        action = capture.parent.name.split('_', 1)[0]
        hp = args.humanpose / f'{action}.humanpose_joints.json'
        if not hp.exists():
            raise FileNotFoundError(hp)
        dest = args.output / 'captures' / action
        dest.mkdir(parents=True)
        report = prepare(capture, dest)
        layout = report['layouts'].get('wrists_shanks_waist', {})
        if layout.get('status') != 'prepared':
            raise RuntimeError(f'{capture}: five-sensor layout unavailable: {layout}')
        package = json.loads(hp.read_text(encoding='utf-8'))
        if len(package.get('bones', [])) != 18 or package.get('coordinateSystem') != 'unity_world_xyz':
            raise RuntimeError(f'{hp}: invalid HumanPose metadata')
        inp = Path(layout['input'])
        out = dest / 'targets_humanpose_calibrated_60.pt'
        pose, tran, shape, _, calibration = retarget_cache(hp, mode='calibrated', calibration_iterations=300)
        data = torch.load(inp, map_location='cpu', weights_only=False)
        timestamps = data['timestamps']
        if data.get('fps') != 60 or (timestamps[1:] - timestamps[:-1]).median().item() < 0.016:
            raise RuntimeError(f'{action}: physical inputs were not resampled to 60 Hz')
        if timestamps[-1] > package['frames'][-1]['time'] + 1e-3:
            raise RuntimeError(f'{action}: IMU timeline exceeds HumanPose timeline')
        from scipy.spatial.transform import Rotation, Slerp
        import numpy as np
        source_t = np.asarray([f['time'] for f in package['frames']], dtype=np.float64)
        target_t = timestamps.numpy().astype(np.float64)
        target_t = np.clip(target_t, source_t[0], source_t[-1])
        pn = pose.numpy()
        p60 = np.stack([Slerp(source_t, Rotation.from_matrix(pn[:, j]))(target_t).as_matrix()
                        for j in range(24)], axis=1)
        tn = tran.numpy()
        t60 = np.stack([np.interp(target_t, source_t, tn[:, k]) for k in range(3)], axis=-1)
        result = {'pose': torch.from_numpy(p60.astype('float32')), 'tran': torch.from_numpy(t60.astype('float32')),
                  'shape': shape.float(), 'timestamps': timestamps.clone(), 'fps': 60,
                  'metadata': {'kind': 'Unity HumanPose rotation retarget pseudo-labels',
                               'independent_ground_truth': False, 'action': action,
                               'humanpose_sha256': hashlib.sha256(hp.read_bytes()).hexdigest(),
                               'raw_imu_sha256': report['raw_imu_sha256'],
                               'mode': 'calibrated', 'calibration': calibration,
                               'source_fps': package.get('fps'), 'target_fps': 60}}
        torch.save(result, out)
        report['humanpose'] = {'path': str(hp), 'sha256': result['metadata']['humanpose_sha256'],
                               'frames': len(package['frames'])}
        report['target'] = str(out)
        (dest / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        catalog['captures'].append({'action': action, 'capture': str(capture), 'frames': int(timestamps.numel()),
                                    'input': str(inp), 'target': str(out), 'raw_imu_sha256': report['raw_imu_sha256'],
                                    'humanpose_sha256': result['metadata']['humanpose_sha256']})
        print(json.dumps(catalog['captures'][-1], ensure_ascii=False), flush=True)
    (args.output / 'catalog.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(json.dumps({'status': 'ok', 'captures': len(catalog['captures']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
