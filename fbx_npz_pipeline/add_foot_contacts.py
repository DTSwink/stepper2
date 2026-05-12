from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from foot_contact import DEFAULT_CONFIG, FootContactConfig, compute_contacts_from_arrays


CONTACT_KEYS = {
    "contact_names",
    "contacts",
    "contact_height_m",
    "contact_speed_mps",
    "contact_closest_point_m",
    "contact_closest_part",
    "contact_lowest_point_m",
    "contact_lowest_part",
    "contact_slide_distance_m",
    "contact_slide_point_prev_m",
    "contact_slide_point_cur_m",
    "contact_slide_part",
    "contact_height_threshold_m",
    "contact_speed_threshold_mps",
}


def load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npz", dir=path.parent) as temp:
        temp_path = Path(temp.name)
    try:
        np.savez_compressed(temp_path, **payload)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def annotate_npz(
    input_npz: Path,
    output_npz: Path,
    config: FootContactConfig = DEFAULT_CONFIG,
    report_json: Path | None = None,
) -> dict:
    payload = load_npz_payload(input_npz)
    for key in CONTACT_KEYS:
        payload.pop(key, None)
    bone_names = [str(x) for x in payload["bone_names"]]
    up_axis = int(payload["axis_up_axis"]) if "axis_up_axis" in payload else 2
    contacts = compute_contacts_from_arrays(
        np.asarray(payload["global_joint_pos"], dtype=np.float32),
        np.asarray(payload["global_matrix"], dtype=np.float32),
        bone_names,
        float(payload["fps"]),
        up_axis,
        config,
    )
    payload.update(contacts)
    write_npz_atomic(output_npz, payload)

    contact_bool = contacts["contacts"]
    summary = {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "frames": int(contact_bool.shape[0]),
        "contact_names": [str(x) for x in contacts["contact_names"]],
        "contact_frame_counts": {
            str(name): int(contact_bool[:, i].sum()) for i, name in enumerate(contacts["contact_names"])
        },
        "contact_ratio": {
            str(name): float(contact_bool[:, i].mean()) for i, name in enumerate(contacts["contact_names"])
        },
        "height_threshold_m": float(config.height_threshold_m),
        "speed_threshold_mps": float(config.horizontal_speed_threshold_mps),
        "min_height_m": contacts["contact_height_m"].min(axis=0).astype(float).tolist(),
        "max_height_m": contacts["contact_height_m"].max(axis=0).astype(float).tolist(),
        "median_height_m": np.median(contacts["contact_height_m"], axis=0).astype(float).tolist(),
        "median_speed_mps": np.median(contacts["contact_speed_mps"], axis=0).astype(float).tolist(),
    }
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def annotate_folder(
    input_folder: Path,
    output_folder: Path,
    config: FootContactConfig = DEFAULT_CONFIG,
    report_folder: Path | None = None,
) -> list[dict]:
    paths = sorted(input_folder.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {input_folder}")
    summaries = []
    for path in paths:
        report = None if report_folder is None else report_folder / f"{path.stem}_contacts.json"
        summaries.append(annotate_npz(path, output_folder / path.name, config, report))
    return summaries


def config_from_args(args: argparse.Namespace) -> FootContactConfig:
    return FootContactConfig(
        foot_length=args.foot_length,
        foot_width=args.foot_width,
        foot_height=args.foot_height,
        toe_length=args.toe_length,
        toe_width=args.toe_width,
        toe_height=args.toe_height,
        sole_vertical_offset=args.sole_vertical_offset,
        position_unit_scale=args.position_unit_scale,
        ground_y=args.ground_y,
        height_threshold_m=args.height_threshold_m,
        horizontal_speed_threshold_mps=args.speed_threshold_mps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add bool left/right foot contact arrays to motion NPZ files.")
    parser.add_argument("input", type=Path, help="Input .npz file or folder.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .npz file or folder. Defaults to in-place.")
    parser.add_argument("--report", type=Path, default=None, help="Optional report .json file or folder.")
    parser.add_argument("--height-threshold-m", type=float, default=DEFAULT_CONFIG.height_threshold_m)
    parser.add_argument("--speed-threshold-mps", type=float, default=DEFAULT_CONFIG.horizontal_speed_threshold_mps)
    parser.add_argument("--position-unit-scale", type=float, default=DEFAULT_CONFIG.position_unit_scale)
    parser.add_argument("--ground-y", type=float, default=DEFAULT_CONFIG.ground_y)
    parser.add_argument("--sole-vertical-offset", type=float, default=DEFAULT_CONFIG.sole_vertical_offset)
    parser.add_argument("--foot-length", type=float, default=DEFAULT_CONFIG.foot_length)
    parser.add_argument("--foot-width", type=float, default=DEFAULT_CONFIG.foot_width)
    parser.add_argument("--foot-height", type=float, default=DEFAULT_CONFIG.foot_height)
    parser.add_argument("--toe-length", type=float, default=DEFAULT_CONFIG.toe_length)
    parser.add_argument("--toe-width", type=float, default=DEFAULT_CONFIG.toe_width)
    parser.add_argument("--toe-height", type=float, default=DEFAULT_CONFIG.toe_height)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output is not None else input_path
    report_path = args.report.resolve() if args.report is not None else None
    config = config_from_args(args)

    if input_path.is_dir():
        summaries = annotate_folder(input_path, output_path, config, report_path)
    else:
        output_npz = output_path
        if output_npz.suffix.lower() != ".npz":
            output_npz = output_npz / input_path.name
        report_json = report_path
        if report_json is not None and report_json.suffix.lower() != ".json":
            report_json = report_json / f"{input_path.stem}_contacts.json"
        summaries = [annotate_npz(input_path, output_npz, config, report_json)]

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
