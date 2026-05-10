from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the contents of a motion NPZ.")
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()

    data = np.load(args.npz)
    print(f"file: {args.npz}")
    for key in sorted(data.files):
        value = data[key]
        print(f"{key}: shape={value.shape} dtype={value.dtype}")
        if key == "bone_names":
            names = [str(x) for x in value[: min(30, len(value))]]
            print("  first bones:", ", ".join(names))


if __name__ == "__main__":
    main()
