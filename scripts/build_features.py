#!/usr/bin/env python
"""Build and cache the windowed feature table.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --window 60 --hop 15 --offset 30
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogload.pipeline import build_feature_table, cache_feature_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=30, help="Window length in seconds")
    parser.add_argument("--hop",    type=float, default=15, help="Hop between windows in seconds")
    parser.add_argument("--offset", type=float, default=30, help="Baseline start offset in seconds")
    args = parser.parse_args()

    print(f"Building feature table: window={args.window}s hop={args.hop}s offset={args.offset}s")
    df   = build_feature_table(args.window, args.hop, args.offset)
    path = cache_feature_table(df, args.window, args.hop, args.offset)

    print(f"\nWindows : {len(df)}")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    print(f"Balance : {(df['label']==1).sum()} cl=1 / {(df['label']==0).sum()} cl=0")
    print(f"Saved   : {path}")


if __name__ == "__main__":
    main()
