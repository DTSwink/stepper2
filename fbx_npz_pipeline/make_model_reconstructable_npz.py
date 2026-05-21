from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


MATRIX_JOINT_KEYS = ("local_matrix", "global_matrix")
ROOT_NAME = "root"
PELVIS_NAME = "pelvis"
STALE_MODEL_KEYS = {
    "ik_reconstructable",
    "ik_reconstruction_version",
    "ik_reconstruction_contract",
    "local_quat_xyzw",
    "fbx_lcl_rotation_euler_xyz",
    "default_lcl_rotation_euler_xyz",
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


def topo_order(parents: np.ndarray) -> list[int]:
    order: list[int] = []
    visiting = np.zeros((parents.shape[0],), dtype=np.bool_)
    visited = np.zeros((parents.shape[0],), dtype=np.bool_)

    def visit(i: int) -> None:
        if visited[i]:
            return
        if visiting[i]:
            raise ValueError("Skeleton parent graph contains a cycle")
        visiting[i] = True
        parent = int(parents[i])
        if parent >= 0:
            visit(parent)
        visiting[i] = False
        visited[i] = True
        order.append(i)

    for joint in range(parents.shape[0]):
        visit(joint)
    return order


def make_matrix(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.zeros(rot.shape[:-2] + (4, 4), dtype=np.float32)
    out[..., :3, :3] = rot.astype(np.float32)
    out[..., 3, :3] = pos.astype(np.float32)
    out[..., 3, 3] = 1.0
    return out


def rotation_6d(rot: np.ndarray) -> np.ndarray:
    return rot[..., :2, :3].reshape(rot.shape[:-2] + (6,)).astype(np.float32)


def raw_local_from_global(
    global_pos: np.ndarray,
    global_rot: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames, joints = global_pos.shape[:2]
    local_pos = np.zeros((frames, joints, 3), dtype=np.float32)
    local_rot = np.zeros((frames, joints, 3, 3), dtype=np.float32)
    for joint in topo_order(parents):
        parent = int(parents[joint])
        if parent < 0:
            local_pos[:, joint] = global_pos[:, joint]
            local_rot[:, joint] = global_rot[:, joint]
            continue
        parent_inv = np.swapaxes(global_rot[:, parent], -1, -2)
        local_pos[:, joint] = np.einsum(
            "tc,tcd->td",
            global_pos[:, joint] - global_pos[:, parent],
            parent_inv,
        )
        local_rot[:, joint] = np.matmul(global_rot[:, joint], parent_inv)
    return local_pos, local_rot


def fk_from_model_locals(
    model_local_pos: np.ndarray,
    local_rot: np.ndarray,
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames, joints = model_local_pos.shape[:2]
    global_pos = np.zeros((frames, joints, 3), dtype=np.float32)
    global_rot = np.zeros((frames, joints, 3, 3), dtype=np.float32)
    for joint in topo_order(parents):
        parent = int(parents[joint])
        if parent < 0:
            global_pos[:, joint] = model_local_pos[:, joint]
            global_rot[:, joint] = local_rot[:, joint]
            continue
        global_rot[:, joint] = np.matmul(local_rot[:, joint], global_rot[:, parent])
        global_pos[:, joint] = (
            np.einsum("tc,tcd->td", model_local_pos[:, joint], global_rot[:, parent])
            + global_pos[:, parent]
        )
    return global_pos, global_rot


def max_norm(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b, axis=-1).max())


def bake_npz(input_npz: Path, output_npz: Path, report_json: Path | None = None) -> dict:
    payload = load_npz_payload(input_npz)
    names = [str(x) for x in payload["bone_names"]]
    parents = np.asarray(payload["parents"], dtype=np.int32)
    if ROOT_NAME not in names:
        raise ValueError(f"{input_npz} does not contain required bone {ROOT_NAME!r}")
    if PELVIS_NAME not in names:
        raise ValueError(f"{input_npz} does not contain required bone {PELVIS_NAME!r}")
    root = names.index(ROOT_NAME)
    pelvis = names.index(PELVIS_NAME)

    source_global_pos = np.asarray(payload["global_joint_pos"], dtype=np.float32)
    source_global_rot = np.asarray(payload["global_matrix"], dtype=np.float32)[..., :3, :3]
    source_local_pos, local_rot = raw_local_from_global(source_global_pos, source_global_rot, parents)

    model_local_pos = np.broadcast_to(source_local_pos[:1], source_local_pos.shape).copy()
    model_local_pos[:, root] = source_local_pos[:, root]
    model_local_pos[:, pelvis] = source_local_pos[:, pelvis]

    model_global_pos, model_global_rot = fk_from_model_locals(model_local_pos, local_rot, parents)
    model_local_matrix = make_matrix(local_rot, model_local_pos)
    model_global_matrix = make_matrix(model_global_rot, model_global_pos)

    source_frame0_delta = max_norm(model_global_pos[:1], source_global_pos[:1])
    source_all_delta = max_norm(model_global_pos, source_global_pos)
    self_pos, self_rot = fk_from_model_locals(model_local_pos, local_rot, parents)
    self_frame0_delta = max_norm(self_pos[:1], model_global_pos[:1])
    self_all_delta = max_norm(self_pos, model_global_pos)

    for key in list(payload):
        if key.startswith("model_") or key in STALE_MODEL_KEYS:
            payload.pop(key, None)

    payload["local_matrix"] = model_local_matrix
    payload["global_matrix"] = model_global_matrix
    payload["local_rotation_6d"] = rotation_6d(local_rot)
    payload["local_translation"] = model_local_pos.astype(np.float32)
    payload["global_joint_pos"] = model_global_pos.astype(np.float32)
    payload["fbx_lcl_translation"] = model_local_pos.astype(np.float32)
    payload["default_lcl_translation"] = model_local_pos[0].astype(np.float32)
    payload["default_local_matrix"] = model_local_matrix[0].astype(np.float32)
    payload["default_global_matrix"] = model_global_matrix[0].astype(np.float32)
    payload["model_reconstructable"] = np.array(True, dtype=np.bool_)
    payload["model_reconstruction_version"] = np.array(1, dtype=np.int32)
    payload["model_reference_frame"] = np.array(0, dtype=np.int32)
    payload["model_reconstruction_contract"] = np.array(
        "row_vector_fk; animated roots; animated pelvis; fixed frame0 offsets for other bones"
    )
    payload["model_source_frame0_max_position_delta"] = np.array(source_frame0_delta, dtype=np.float32)
    payload["model_source_all_max_position_delta"] = np.array(source_all_delta, dtype=np.float32)
    payload["model_self_frame0_max_position_delta"] = np.array(self_frame0_delta, dtype=np.float32)
    payload["model_self_all_max_position_delta"] = np.array(self_all_delta, dtype=np.float32)

    write_npz_atomic(output_npz, payload)
    summary = {
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "frames": int(source_global_pos.shape[0]),
        "bones": int(source_global_pos.shape[1]),
        "root": ROOT_NAME,
        "pelvis": PELVIS_NAME,
        "source_frame0_max_position_delta": source_frame0_delta,
        "source_all_max_position_delta": source_all_delta,
        "self_frame0_max_position_delta": self_frame0_delta,
        "self_all_max_position_delta": self_all_delta,
        "contract": str(payload["model_reconstruction_contract"]),
    }
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def bake_folder(input_folder: Path, output_folder: Path, report_folder: Path | None = None) -> list[dict]:
    paths = sorted(input_folder.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {input_folder}")
    summaries = []
    for path in paths:
        report = None if report_folder is None else report_folder / f"{path.stem}_model_reconstructable.json"
        summaries.append(bake_npz(path, output_folder / path.name, report))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake motion NPZ files to the fixed-offset model FK contract.")
    parser.add_argument("input", type=Path, help="Input .npz file or folder.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .npz file or folder. Defaults to in-place.")
    parser.add_argument("--report", type=Path, default=None, help="Optional report .json file or folder.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output is not None else input_path
    report_path = args.report.resolve() if args.report is not None else None
    if input_path.is_dir():
        summaries = bake_folder(input_path, output_path, report_path)
    else:
        output_npz = output_path
        if output_npz.suffix.lower() != ".npz":
            output_npz = output_npz / input_path.name
        report_json = report_path
        if report_json is not None and report_json.suffix.lower() != ".json":
            report_json = report_json / f"{input_path.stem}_model_reconstructable.json"
        summaries = [bake_npz(input_path, output_npz, report_json)]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
