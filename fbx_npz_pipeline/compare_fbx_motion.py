from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import fbx
    import FbxCommon  # type: ignore
except ImportError:
    sdk_samples = (
        Path(__file__).resolve().parents[1]
        / ".tools"
        / "fbx_python_sdk_2020.3.4"
        / "samples"
    )
    if sdk_samples.exists():
        sys.path.insert(0, str(sdk_samples))
    import fbx
    import FbxCommon  # type: ignore

from fbx_to_npz import collect_skeleton_nodes, fbx_vec3_to_np, load_scene, make_sample_times


def eval_curve(prop, layer, axis_name: str, time: "fbx.FbxTime", default_value: float) -> float:
    if layer is None:
        return float(default_value)
    curve = prop.GetCurve(layer, axis_name, False)
    if curve is None:
        return float(default_value)
    value = curve.Evaluate(time)
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def extract_for_compare(path: Path) -> dict:
    manager, scene = load_scene(path)
    try:
        nodes = collect_skeleton_nodes(scene)
        stack = scene.GetCurrentAnimationStack()
        layer = None
        if stack is not None:
            layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
        times, fps_value, start_frame, stop_frame = make_sample_times(scene)

        names = [node.GetName() for node in nodes]
        frames = len(times)
        joints = len(nodes)
        pos = np.zeros((frames, joints, 3), dtype=np.float64)
        lcl_t = np.zeros((frames, joints, 3), dtype=np.float64)
        lcl_r = np.zeros((frames, joints, 3), dtype=np.float64)
        lcl_s = np.zeros((frames, joints, 3), dtype=np.float64)

        for ti, time in enumerate(times):
            for ji, node in enumerate(nodes):
                pos[ti, ji] = fbx_vec3_to_np(node.EvaluateGlobalTransform(time).GetT())
                defaults = [
                    node.LclTranslation.Get(),
                    node.LclRotation.Get(),
                    node.LclScaling.Get(),
                ]
                for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                    lcl_t[ti, ji, axis_index] = eval_curve(
                        node.LclTranslation, layer, axis_name, time, defaults[0][axis_index]
                    )
                    lcl_r[ti, ji, axis_index] = eval_curve(
                        node.LclRotation, layer, axis_name, time, defaults[1][axis_index]
                    )
                    lcl_s[ti, ji, axis_index] = eval_curve(
                        node.LclScaling, layer, axis_name, time, defaults[2][axis_index]
                    )

        return {
            "names": names,
            "fps": fps_value,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
            "global_joint_pos": pos,
            "fbx_lcl_translation": lcl_t,
            "fbx_lcl_rotation_euler_xyz": lcl_r,
            "fbx_lcl_scale": lcl_s,
        }
    finally:
        manager.Destroy()


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two FBX animations by SDK-evaluated motion.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    ref = extract_for_compare(args.reference.resolve())
    cand = extract_for_compare(args.candidate.resolve())
    if ref["names"] != cand["names"]:
        raise RuntimeError("Skeleton names/order differ between FBX files.")

    metrics = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "frames_reference": int(ref["global_joint_pos"].shape[0]),
        "frames_candidate": int(cand["global_joint_pos"].shape[0]),
        "bones": len(ref["names"]),
        "fps_reference": ref["fps"],
        "fps_candidate": cand["fps"],
        "max_abs_global_joint_pos": max_abs(ref["global_joint_pos"], cand["global_joint_pos"]),
        "max_abs_fbx_lcl_translation": max_abs(ref["fbx_lcl_translation"], cand["fbx_lcl_translation"]),
        "max_abs_fbx_lcl_rotation_euler_xyz": max_abs(
            ref["fbx_lcl_rotation_euler_xyz"], cand["fbx_lcl_rotation_euler_xyz"]
        ),
        "max_abs_fbx_lcl_scale": max_abs(ref["fbx_lcl_scale"], cand["fbx_lcl_scale"]),
    }
    metrics["passed"] = all(
        metrics[key] <= args.tolerance
        for key in (
            "max_abs_global_joint_pos",
            "max_abs_fbx_lcl_translation",
            "max_abs_fbx_lcl_rotation_euler_xyz",
            "max_abs_fbx_lcl_scale",
        )
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
