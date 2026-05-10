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
    # Autodesk ships FbxCommon.py beside the Python SDK samples. The setup script
    # copies that file into this folder, but this fallback keeps direct runs usable.
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


def fbx_matrix_to_np(matrix: "fbx.FbxAMatrix") -> np.ndarray:
    out = np.empty((4, 4), dtype=np.float64)
    for r in range(4):
        for c in range(4):
            out[r, c] = matrix.Get(r, c)
    return out


def fbx_vec3_to_np(vec: "fbx.FbxDouble3") -> np.ndarray:
    return np.array([vec[0], vec[1], vec[2]], dtype=np.float64)


def fbx_quat_to_np(quat: "fbx.FbxQuaternion") -> np.ndarray:
    return np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    # Autodesk FBX matrices use the SDK's row-vector layout. Taking the first
    # two basis rows gives the same information as first-two-columns in a
    # conventional column-vector rotation matrix.
    return matrix[:2, :3].reshape(6)


def traverse_nodes(node: "fbx.FbxNode", nodes: list["fbx.FbxNode"]) -> None:
    nodes.append(node)
    for i in range(node.GetChildCount()):
        traverse_nodes(node.GetChild(i), nodes)


def is_skeleton_node(node: "fbx.FbxNode") -> bool:
    attr = node.GetNodeAttribute()
    if attr is None:
        return False
    return attr.GetAttributeType() == fbx.FbxNodeAttribute.EType.eSkeleton


def collect_skeleton_nodes(scene: "fbx.FbxScene") -> list["fbx.FbxNode"]:
    root = scene.GetRootNode()
    all_nodes: list[fbx.FbxNode] = []
    for i in range(root.GetChildCount()):
        traverse_nodes(root.GetChild(i), all_nodes)
    skeleton_nodes = [node for node in all_nodes if is_skeleton_node(node)]
    if skeleton_nodes:
        return skeleton_nodes

    # Fallback for unusual exports: keep animated transform nodes if no explicit
    # skeleton attribute exists. This is intentionally conservative.
    animated = []
    for node in all_nodes:
        if any(
            node.LclTranslation.GetCurve(None, axis)
            or node.LclRotation.GetCurve(None, axis)
            or node.LclScaling.GetCurve(None, axis)
            for axis in ("X", "Y", "Z")
        ):
            animated.append(node)
    return animated


def get_parent_indices(nodes: list["fbx.FbxNode"]) -> np.ndarray:
    index_by_uid = {node.GetUniqueID(): i for i, node in enumerate(nodes)}
    parents = np.full((len(nodes),), -1, dtype=np.int32)
    for i, node in enumerate(nodes):
        parent = node.GetParent()
        while parent is not None:
            parent_index = index_by_uid.get(parent.GetUniqueID())
            if parent_index is not None:
                parents[i] = parent_index
                break
            parent = parent.GetParent()
    return parents


def get_animation_time_span(scene: "fbx.FbxScene") -> tuple["fbx.FbxTime", "fbx.FbxTime"]:
    stack_count = scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
    if stack_count > 0:
        stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        scene.SetCurrentAnimationStack(stack)
        span = stack.GetLocalTimeSpan()
        return span.GetStart(), span.GetStop()

    span = scene.GetGlobalSettings().GetTimelineDefaultTimeSpan()
    return span.GetStart(), span.GetStop()


def get_scene_fps(scene: "fbx.FbxScene") -> float:
    time_mode = scene.GetGlobalSettings().GetTimeMode()
    fps = fbx.FbxTime.GetFrameRate(time_mode)
    return float(fps if fps > 0 else 30.0)


def make_sample_times(scene: "fbx.FbxScene") -> tuple[list["fbx.FbxTime"], float, int, int]:
    fps_value = get_scene_fps(scene)
    start, stop = get_animation_time_span(scene)
    start_frame = int(round(start.GetSecondDouble() * fps_value))
    stop_frame = int(round(stop.GetSecondDouble() * fps_value))
    if stop_frame < start_frame:
        stop_frame = start_frame

    times = []
    for frame in range(start_frame, stop_frame + 1):
        t = fbx.FbxTime()
        t.SetSecondDouble(frame / fps_value)
        times.append(t)
    return times, fps_value, start_frame, stop_frame


def get_axis_system(scene: "fbx.FbxScene") -> dict[str, int]:
    def enum_int(value) -> int:
        return int(getattr(value, "value", value))

    axis = scene.GetGlobalSettings().GetAxisSystem()
    up_axis, up_sign = axis.GetUpVector()
    front_axis, front_sign = axis.GetFrontVector()
    coord_axis = axis.GetCoorSystem()
    return {
        "up_axis": enum_int(up_axis),
        "up_sign": enum_int(up_sign),
        "front_axis": enum_int(front_axis),
        "front_sign": enum_int(front_sign),
        "coord_axis": enum_int(coord_axis),
    }


def get_system_unit(scene: "fbx.FbxScene") -> dict[str, float | str]:
    unit = scene.GetGlobalSettings().GetSystemUnit()
    return {
        "scale_factor_cm": float(unit.GetScaleFactor()),
        "multiplier": float(unit.GetMultiplier()),
    }


def load_scene(path: Path) -> tuple["fbx.FbxManager", "fbx.FbxScene"]:
    manager, scene = FbxCommon.InitializeSdkObjects()
    if not FbxCommon.LoadScene(manager, scene, str(path)):
        raise RuntimeError(f"Autodesk FBX SDK could not load: {path}")
    return manager, scene


def extract_fbx_to_npz(input_fbx: Path, output_npz: Path, report_json: Path | None) -> dict:
    manager, scene = load_scene(input_fbx)
    try:
        nodes = collect_skeleton_nodes(scene)
        if not nodes:
            raise RuntimeError("No skeleton or animated transform nodes found.")

        parents = get_parent_indices(nodes)
        bone_names = np.array([node.GetName() for node in nodes])
        bone_uids = np.array([node.GetUniqueID() for node in nodes], dtype=np.int64)

        times, fps_value, start_frame, stop_frame = make_sample_times(scene)
        frame_count = len(times)
        joint_count = len(nodes)

        global_mats = np.zeros((frame_count, joint_count, 4, 4), dtype=np.float32)
        local_mats = np.zeros((frame_count, joint_count, 4, 4), dtype=np.float32)
        local_quat_xyzw = np.zeros((frame_count, joint_count, 4), dtype=np.float32)
        local_rot_6d = np.zeros((frame_count, joint_count, 6), dtype=np.float32)
        local_translation = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
        local_scale = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
        global_joint_pos = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
        fbx_lcl_translation = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
        fbx_lcl_rotation_euler_xyz = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
        fbx_lcl_scale = np.zeros((frame_count, joint_count, 3), dtype=np.float32)

        stack = scene.GetCurrentAnimationStack()
        layer = None
        if stack is not None:
            layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)

        def eval_lcl_property(prop, time, defaults):
            values = np.array([defaults[0], defaults[1], defaults[2]], dtype=np.float64)
            if layer is None:
                return values
            for axis_index, axis_name in enumerate(("X", "Y", "Z")):
                curve = prop.GetCurve(layer, axis_name, False)
                if curve is not None:
                    value = curve.Evaluate(time)
                    if isinstance(value, tuple):
                        value = value[0]
                    values[axis_index] = float(value)
            return values

        for ti, time in enumerate(times):
            evaluated_global = [node.EvaluateGlobalTransform(time) for node in nodes]
            for ji, node in enumerate(nodes):
                global_np = fbx_matrix_to_np(evaluated_global[ji])
                parent_idx = parents[ji]
                if parent_idx >= 0:
                    parent_inv = evaluated_global[parent_idx].Inverse()
                    local_fbx = parent_inv * evaluated_global[ji]
                else:
                    local_fbx = evaluated_global[ji]

                local_np = fbx_matrix_to_np(local_fbx)
                q = local_fbx.GetQ()
                t = local_fbx.GetT()
                s = local_fbx.GetS()

                global_mats[ti, ji] = global_np
                local_mats[ti, ji] = local_np
                local_quat_xyzw[ti, ji] = fbx_quat_to_np(q)
                local_rot_6d[ti, ji] = matrix_to_rotation_6d(local_np)
                local_translation[ti, ji] = fbx_vec3_to_np(t)
                local_scale[ti, ji] = fbx_vec3_to_np(s)
                global_joint_pos[ti, ji] = fbx_vec3_to_np(evaluated_global[ji].GetT())
                fbx_lcl_translation[ti, ji] = eval_lcl_property(
                    node.LclTranslation, time, node.LclTranslation.Get()
                )
                fbx_lcl_rotation_euler_xyz[ti, ji] = eval_lcl_property(
                    node.LclRotation, time, node.LclRotation.Get()
                )
                fbx_lcl_scale[ti, ji] = eval_lcl_property(
                    node.LclScaling, time, node.LclScaling.Get()
                )

        default_local_mats = np.zeros((joint_count, 4, 4), dtype=np.float32)
        default_global_mats = np.zeros((joint_count, 4, 4), dtype=np.float32)
        default_lcl_translation = np.zeros((joint_count, 3), dtype=np.float32)
        default_lcl_rotation_euler_xyz = np.zeros((joint_count, 3), dtype=np.float32)
        default_lcl_scale = np.zeros((joint_count, 3), dtype=np.float32)
        zero = fbx.FbxTime()
        for ji, node in enumerate(nodes):
            default_global_mats[ji] = fbx_matrix_to_np(node.EvaluateGlobalTransform(zero))
            default_local_mats[ji] = fbx_matrix_to_np(node.EvaluateLocalTransform(zero))
            default_lcl_translation[ji] = fbx_vec3_to_np(node.LclTranslation.Get())
            default_lcl_rotation_euler_xyz[ji] = fbx_vec3_to_np(node.LclRotation.Get())
            default_lcl_scale[ji] = fbx_vec3_to_np(node.LclScaling.Get())

        output_npz.parent.mkdir(parents=True, exist_ok=True)
        axis_system = get_axis_system(scene)
        system_unit = get_system_unit(scene)
        np.savez_compressed(
            output_npz,
            source_fbx=np.array(str(input_fbx)),
            axis_up_axis=np.array(axis_system["up_axis"], dtype=np.int32),
            axis_up_sign=np.array(axis_system["up_sign"], dtype=np.int32),
            axis_front_axis=np.array(axis_system["front_axis"], dtype=np.int32),
            axis_front_sign=np.array(axis_system["front_sign"], dtype=np.int32),
            axis_coord_axis=np.array(axis_system["coord_axis"], dtype=np.int32),
            system_unit_scale_factor_cm=np.array(system_unit["scale_factor_cm"], dtype=np.float32),
            system_unit_multiplier=np.array(system_unit["multiplier"], dtype=np.float32),
            bone_names=bone_names,
            bone_uids=bone_uids,
            parents=parents,
            fps=np.array(fps_value, dtype=np.float32),
            frame_start=np.array(start_frame, dtype=np.int32),
            frame_end=np.array(stop_frame, dtype=np.int32),
            frame_count=np.array(frame_count, dtype=np.int32),
            local_matrix=local_mats,
            global_matrix=global_mats,
            local_quat_xyzw=local_quat_xyzw,
            local_rotation_6d=local_rot_6d,
            local_translation=local_translation,
            local_scale=local_scale,
            global_joint_pos=global_joint_pos,
            fbx_lcl_translation=fbx_lcl_translation,
            fbx_lcl_rotation_euler_xyz=fbx_lcl_rotation_euler_xyz,
            fbx_lcl_scale=fbx_lcl_scale,
            default_local_matrix=default_local_mats,
            default_global_matrix=default_global_mats,
            default_lcl_translation=default_lcl_translation,
            default_lcl_rotation_euler_xyz=default_lcl_rotation_euler_xyz,
            default_lcl_scale=default_lcl_scale,
        )

        report = {
            "source_fbx": str(input_fbx),
            "output_npz": str(output_npz),
            "frames": frame_count,
            "frame_start": start_frame,
            "frame_end": stop_frame,
            "fps": fps_value,
            "bones": joint_count,
            "root_bones": [bone_names[i].item() for i, p in enumerate(parents) if p < 0],
            "axis_system": axis_system,
            "system_unit": system_unit,
            "first_bones": bone_names[: min(20, joint_count)].tolist(),
            "arrays": {
                "local_matrix": list(local_mats.shape),
                "global_matrix": list(global_mats.shape),
                "local_quat_xyzw": list(local_quat_xyzw.shape),
                "local_rotation_6d": list(local_rot_6d.shape),
                "global_joint_pos": list(global_joint_pos.shape),
                "fbx_lcl_translation": list(fbx_lcl_translation.shape),
                "fbx_lcl_rotation_euler_xyz": list(fbx_lcl_rotation_euler_xyz.shape),
                "fbx_lcl_scale": list(fbx_lcl_scale.shape),
            },
        }

        if report_json is not None:
            report_json.parent.mkdir(parents=True, exist_ok=True)
            report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return report
    finally:
        manager.Destroy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract an FBX skeleton animation to an NPZ file using Autodesk FBX SDK."
    )
    parser.add_argument("input_fbx", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    input_fbx = args.input_fbx.resolve()
    output = args.output
    if output is None:
        output = Path("data") / "npz" / f"{input_fbx.stem}.npz"
    report = args.report
    if report is None:
        report = Path("data") / "reports" / f"{input_fbx.stem}.json"

    summary = extract_fbx_to_npz(input_fbx, output.resolve(), report.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
