# FBX to NPZ Pipeline

This folder is for the game-facing motion data path:

```text
UEFN/Cascadeur FBX -> Autodesk FBX SDK sampling -> NPZ tensors
```

The converter intentionally does not go through Blender or BVH. It evaluates the
FBX scene with Autodesk's SDK, samples every frame, and writes redundant arrays so
the later `NPZ -> FBX` round trip can be validated before training.

## Local Setup

The project-local tools are installed under:

```text
.tools/python310
.tools/fbx_python_sdk_2020.3.4
```

Run the converter with:

```powershell
.\.tools\python310\python.exe .\fbx_npz_pipeline\fbx_to_npz.py C:\path\to\anim.fbx
```

or with the wrapper:

```powershell
.\fbx_npz_pipeline\convert_fbx_to_npz.ps1 C:\path\to\anim.fbx
```

The default outputs are:

```text
data/npz/<clip>.npz
data/reports/<clip>.json
```

## Saved Arrays

- `local_matrix [T, J, 4, 4]`: evaluated local bone transforms relative to the
  collected skeleton parent.
- `global_matrix [T, J, 4, 4]`: evaluated global bone transforms.
- `local_quat_xyzw [T, J, 4]`: local rotation quaternions.
- `local_rotation_6d [T, J, 6]`: first two rotation columns for NN training.
- `local_translation [T, J, 3]`: local translation, including root motion.
- `global_joint_pos [T, J, 3]`: global joint positions for losses/debugging.

For game fidelity, first validate:

```text
FBX -> NPZ -> FBX
```

in Unreal/Cascadeur before training a model.

## Viewing NPZ Files

Create a self-contained HTML skeleton viewer:

```powershell
.\fbx_npz_pipeline\view_npz.ps1 .\data\npz\testcasc.npz
```

The viewer is for training-data sanity checks. It renders `global_joint_pos`
from the NPZ directly, so it avoids Blender/BVH bone-axis issues, but the final
authority for game fidelity should still be the later `NPZ -> FBX -> Unreal`
round trip.

## Coordinate Conventions

The raw NPZ keeps the FBX SDK-evaluated transforms untouched for round-tripping.
Training, contact generation, and viewers canonicalize only at read time:

- Cascadeur exports in this project report Y-up, so the data is used directly.
- UE5 exports report Z-up, so it is mapped to the training convention with
  `canonical = [source_x, source_z, -source_y]`.

Rotation matrices use the FBX row-vector convention, so the same UE5 conversion
is applied as `R_canonical = P^-1 R_source P`. Foot and hand collider axes are
chosen from the actual bone basis and nearby anatomy instead of assuming one
fixed Cascadeur bone-axis layout.

## Foot Contact Metrics

Contact generation stores both height and slide diagnostics, but they are not
the same point calculation:

- `contact_height_m` and `contact_lowest_point_m` come from the absolute lowest
  point on either the foot collider or toe collider. This is the value used for
  contact height, hovering, and penetration checks.
- `contact_speed_mps` comes from `contact_slide_distance_m`. That slide distance
  is the minimum 2D ground-plane displacement over every same-local-point on the
  continuous foot sole rectangle and toe sole rectangle, then the smaller foot
  vs toe result is used. It is not based on the lowest collider point.

## Rebuilding FBX Files

Rebuild an FBX animation from an NPZ using a matching template/reference FBX:

```powershell
.\fbx_npz_pipeline\convert_npz_to_fbx.ps1 `
  .\data\npz\testcasc.npz `
  .\data\fbx\testcasc.fbx `
  .\data\roundtrip_fbx\testcasc_roundtrip.fbx
```

The template FBX supplies the exact skeleton, mesh, bind pose, node settings,
and FBX-specific rotation metadata. The NPZ supplies the baked animation curves.

Compare two FBX animations through Autodesk's evaluator:

```powershell
.\.tools\python310\python.exe .\fbx_npz_pipeline\compare_fbx_motion.py `
  .\data\fbx\testcasc.fbx `
  .\data\roundtrip_fbx\testcasc_roundtrip.fbx `
  --report .\data\reports\testcasc_roundtrip_compare.json
```

For `testcasc`, the current round trip reports zero error for global joint
positions and raw FBX `LclTranslation`, `LclRotation`, and `LclScaling` curves.
