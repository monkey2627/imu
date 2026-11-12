#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--shape-store", type=Path, required=True)
    parser.add_argument("--pose-store", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    from mobileposer.line9.data import load_identity_records
    from mobileposer.line9.pose_store import build_pose_store
    from mobileposer.line9.store import build_store

    records = load_identity_records(args.data, args.audit, ("train", "validation"))
    all_records = records["train"] + records["validation"]
    build_store(all_records, args.shape_store, workers=args.workers, shaped=True)
    build_pose_store(all_records, args.shape_store, args.pose_store, workers=args.workers)
    print(
        f"line14 stores complete: train={len(records['train'])} "
        f"validation={len(records['validation'])}"
    )


if __name__ == "__main__":
    main()
