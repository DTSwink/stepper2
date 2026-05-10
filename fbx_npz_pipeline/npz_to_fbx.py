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

from fbx_to_npz import collect_skeleton_nodes, fbx_vec3_to_np, load_scene


AXES = ("X", "Y", "Z")


def get_or_create_base_layer(scene: "fbx.FbxScene") -> tuple["fbx.FbxAnimStack", "fbx.FbxAnimLayer"]:
    stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
    if stack is None:
        stack = fbx.FbxAnimStack.Create(scene, "NPZ_Roundtrip")
    scene.SetCurrentAnimationStack(stack)

    layer = stack.GetMember(fbx.FbxCriteria.ObjectType(fbx.FbxAnimLayer.ClassId), 0)
    if layer is None:
        layer = fbx.FbxAnimLayer.Create(scene, "Base Layer")
        stack.AddMember(layer)
    return stack, layer


def set_time_span(stack: "fbx.FbxAnimStack", frame_count: int, fps_value: float) -> None:
    start = fbx.FbxTime()
    stop = fbx.FbxTime()
    start.SetSecondDouble(0.0)
    stop.SetSecondDouble((frame_count - 1) / fps_value if frame_count > 1 else 0.0)
    span = fbx.FbxTimeSpan()
    span.SetStart(start)
    span.SetStop(stop)
    stack.SetLocalTimeSpan(span)
    stack.SetReferenceTimeSpan(span)


def set_curve_keys(prop, layer: "fbx.FbxAnimLayer", values: np.ndarray, fps_value: float) -> None:
    for axis_index, axis_name in enumerate(AXES):
        curve = prop.GetCurve(layer, axis_name, True)
        curve.KeyModifyBegin()
        curve.KeyClear()
        for frame_index, value in enumerate(values[:, axis_index]):
            time = fbx.FbxTime()
            time.SetSecondDouble(frame_index / fps_value)
            key_index = curve.KeyAdd(time)[0]
            curve.KeySetValue(key_index, float(value))
            curve.KeySetInterpolation(
                key_index,
                fbx.FbxAnimCurveDef.EInterpolationType.eInterpolationLinear,
            )
        curve.KeyModifyEnd()


def set_node_defaults(node: "fbx.FbxNode", t: np.ndarray, r: np.ndarray, s: np.ndarray) -> None:
    node.LclTranslation.Set(fbx.FbxDouble3(float(t[0]), float(t[1]), float(t[2])))
    node.LclRotation.Set(fbx.FbxDouble3(float(r[0]), float(r[1]), float(r[2])))
    node.LclScaling.Set(fbx.FbxDouble3(float(s[0]), float(s[1]), float(s[2])))


def save_scene_binary(manager: "fbx.FbxManager", scene: "fbx.FbxScene", output_fbx: Path) -> None:
    output_fbx.parent.mkdir(parents=True, exist_ok=True)
    writer_format = manager.GetIOPluginRegistry().GetNativeWriterFormat()
    if not FbxCommon.SaveScene(manager, scene, str(output_fbx), writer_format):
        raise RuntimeError(f"Autodesk FBX SDK could not save: {output_fbx}")


def npz_to_fbx(npz_path: Path, template_fbx: Path, output_fbx: Path, report_json: Path | None) -> dict:
    data = np.load(npz_path)
    required = ("fbx_lcl_translation", "fbx_lcl_rotation_euler_xyz", "fbx_lcl_scale")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise RuntimeError(
            "NPZ is missing raw FBX Lcl arrays. Regenerate it with fbx_to_npz.py. "
            f"Missing: {', '.join(missing)}"
        )

    bone_names = [str(x) for x in data["bone_names"]]
    lcl_t = np.asarray(data["fbx_lcl_translation"], dtype=np.float64)
    lcl_r = np.asarray(data["fbx_lcl_rotation_euler_xyz"], dtype=np.float64)
    lcl_s = np.asarray(data["fbx_lcl_scale"], dtype=np.float64)
    fps_value = float(data["fps"])
    frame_count = int(data["frame_count"])

    if lcl_t.shape != (frame_count, len(bone_names), 3):
        raise RuntimeError(f"Unexpected fbx_lcl_translation shape: {lcl_t.shape}")

    manager, scene = load_scene(template_fbx)
    try:
        nodes = collect_skeleton_nodes(scene)
        template_names = [node.GetName() for node in nodes]
        if template_names != bone_names:
            mismatch = [
                (i, a, b)
                for i, (a, b) in enumerate(zip(template_names, bone_names))
                if a != b
            ][:10]
            raise RuntimeError(
                "Template skeleton order/names do not match NPZ. "
                f"template_bones={len(template_names)} npz_bones={len(bone_names)} "
                f"first_mismatches={mismatch}"
            )

        stack, layer = get_or_create_base_layer(scene)
        set_time_span(stack, frame_count, fps_value)

        for bone_index, node in enumerate(nodes):
            set_node_defaults(node, lcl_t[0, bone_index], lcl_r[0, bone_index], lcl_s[0, bone_index])
            set_curve_keys(node.LclTranslation, layer, lcl_t[:, bone_index, :], fps_value)
            set_curve_keys(node.LclRotation, layer, lcl_r[:, bone_index, :], fps_value)
            set_curve_keys(node.LclScaling, layer, lcl_s[:, bone_index, :], fps_value)

        save_scene_binary(manager, scene, output_fbx)

        report = {
            "source_npz": str(npz_path),
            "template_fbx": str(template_fbx),
            "output_fbx": str(output_fbx),
            "frames": frame_count,
            "fps": fps_value,
            "bones": len(bone_names),
            "anim_stack": stack.GetName(),
            "anim_layer": layer.GetName(),
            "first_bones": bone_names[: min(20, len(bone_names))],
        }
        if report_json is not None:
            report_json.parent.mkdir(parents=True, exist_ok=True)
            report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        manager.Destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild an FBX animation from a motion NPZ.")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    output = args.output
    if output is None:
        output = Path("data") / "roundtrip_fbx" / f"{npz_path.stem}_roundtrip.fbx"
    report = args.report
    if report is None:
        report = Path("data") / "reports" / f"{npz_path.stem}_npz_to_fbx.json"

    summary = npz_to_fbx(npz_path, args.template.resolve(), output.resolve(), report.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
