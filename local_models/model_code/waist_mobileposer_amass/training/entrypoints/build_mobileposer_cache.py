#!/usr/bin/env python3
from mobileposer.data import PoseDataset


if __name__ == "__main__":
    dataset = PoseDataset(fold="train")
    print(f"cached_windows={len(dataset)}")
