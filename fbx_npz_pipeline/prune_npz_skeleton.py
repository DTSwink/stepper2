from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HELPER_PREFIXES = ("ik_", "weapon_")
DETAIL_TOKENS = ("thumb", "index", "middle", "ring", "pinky", "metacarpal")


def is_helper_name(name: str) -> bool:
    return (
        "_twist_" in name
        or any(name.startswith(prefix) for prefix in HELPER_PREFIXES)
        or name == "attach"
    )


def is_detail_name(name: str) -> bool:
    return any(token in name for token in DETAIL_TOKENS)


def should_keep_name(name: str) -> bool:
    return not is_helper_name(name) and not is_detail_name(name)


def remap_parents(parents: np.ndarray, keep: list[int]) -> np.ndarray:
    full_to_new = {full_i: new_i for new_i, full_i in enumerate(keep)}
    remapped = np.full((len(keep),), -1, dtype=np.int32)
    for new_i, full_i in enumerate(keep):
        parent = int(parents[full_i])
        while parent >= 0 and parent not in full_to_new:
            parent = int(parents[parent])
        if parent >= 0:
            remapped[new_i] = full_to_new[parent]
    return remapped


def prune_array(name: str, array: np.ndarray, keep: np.ndarray, joint_count: int) -> np.ndarray:
    if name in {"bone_names", "bone_uids", "parents"}:
        raise ValueError(f"{name} is handled separately")
    if array.ndim >= 1 and array.shape[0] == joint_count:
        return np.take(array, keep, axis=0)
    if array.ndim >= 2 and array.shape[1] == joint_count:
        return np.take(array, keep, axis=1)
    return array


def prune_npz(input_npz: Path, output_npz: Path, report_json: Path | None = None) -> dict:
    with np.load(input_npz, allow_pickle=False) as data:
        names = [str(x) for x in data["bone_names"]]
        parents = np.asarray(data["parents"], dtype=np.int32)
        keep_list = [i for i, name in enumerate(names) if should_keep_name(name)]
        if "root" not in [names[i] for i in keep_list]:
            raise ValueError("Pruned skeleton must keep root for root-motion features.")
        keep = np.asarray(keep_list, dtype=np.int64)
        output = {}
        for key in data.files:
            array = data[key]
            if key == "bone_names":
                output[key] = array[keep]
            elif key == "bone_uids":
                output[key] = array[keep]
            elif key == "parents":
                output[key] = remap_parents(parents, keep_list)
            else:
                output[key] = prune_array(key, array, keep, len(names))

        removed = [name for i, name in enumerate(names) if i not in set(keep_list)]
        summary = {
            "input_npz": str(input_npz),
            "output_npz": str(output_npz),
            "original_bones": len(names),
            "kept_bones": len(keep_list),
            "removed_bones": len(removed),
            "kept_names": [names[i] for i in keep_list],
            "removed_names": removed,
        }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **output)
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prune_folder(input_folder: Path, output_folder: Path, report_folder: Path | None = None) -> list[dict]:
    paths = sorted(input_folder.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {input_folder}")
    summaries = []
    for path in paths:
        report = None if report_folder is None else report_folder / f"{path.stem}_pruned.json"
        summaries.append(prune_npz(path, output_folder / path.name, report))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip helper/detail bones from motion NPZ files.")
    parser.add_argument("input", type=Path, help="Input .npz file or folder.")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/npz_final"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/npz_final"))
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else None
    if input_path.is_dir():
        summaries = prune_folder(input_path, output_path, report_path)
    else:
        output_npz = output_path
        if output_npz.suffix.lower() != ".npz":
            output_npz = output_npz / input_path.name
        report_json = None
        if report_path is not None:
            report_json = report_path
            if report_json.suffix.lower() != ".json":
                report_json = report_json / f"{input_path.stem}_pruned.json"
        summaries = [prune_npz(input_path, output_npz, report_json)]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
