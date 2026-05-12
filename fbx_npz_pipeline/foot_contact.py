from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootContactConfig:
    # Hard-set from training/model_viewer_settings.json on 2026-05-12.
    # Values are meters, after the FBX centimeter data is scaled by 0.01.
    foot_length: float = 0.150
    foot_width: float = 0.110
    foot_height: float = 0.051
    toe_length: float = 0.065
    toe_width: float = 0.110
    toe_height: float = 0.050
    sole_vertical_offset: float = -0.006
    position_unit_scale: float = 0.01
    ground_y: float = 0.0
    height_threshold_m: float = 0.025
    horizontal_speed_threshold_mps: float = 0.350


DEFAULT_CONFIG = FootContactConfig()
CONTACT_NAMES = np.asarray(["contactL", "contactR"])


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    length = np.linalg.norm(axis, axis=-1, keepdims=True)
    return axis / np.maximum(length, 1e-8)


def canonicalize_positions(positions: np.ndarray, up_axis: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if int(up_axis) == 3:
        return positions[..., [1, 2, 0]].copy()
    return positions.copy()


def canonicalize_rotations(rotations: np.ndarray, up_axis: int) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float32)
    if int(up_axis) != 3:
        return rotations.copy()
    # Same row-vector convention as training/model_viewer_app.py:
    # source Z-up -> canonical Y-up, with x=source y, y=source z, z=source x.
    return rotations[..., :, [1, 2, 0]].copy()


def box_lowest_point_signed_height(
    center: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_z: np.ndarray,
    dims: tuple[float, float, float] | np.ndarray,
    ground_y: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return the absolute lowest point on an oriented box and its signed ground height."""

    center = np.asarray(center, dtype=np.float32)
    axes = np.stack(
        (_normalize_axis(axis_x), _normalize_axis(axis_y), _normalize_axis(axis_z)),
        axis=0,
    )
    half_dims = np.asarray(dims, dtype=np.float32) * 0.5
    signs = np.where(axes[:, 1] > 0.0, -1.0, 1.0).astype(np.float32)
    point = center + np.sum(axes * (signs * half_dims)[:, None], axis=0)
    height = float(point[1] - float(ground_y))
    return point.astype(np.float32), height


def box_closest_point_to_ground(
    center: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_z: np.ndarray,
    dims: tuple[float, float, float] | np.ndarray,
    ground_y: float = 0.0,
) -> tuple[np.ndarray, float]:
    return box_lowest_point_signed_height(center, axis_x, axis_y, axis_z, dims, ground_y)


def _project_xz(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec[..., [0, 2]]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _rect_least_squares_minimum(
    a_matrix: np.ndarray,
    b_vec: np.ndarray,
    u_limit: float,
    v_limit: float,
) -> tuple[float, float, float]:
    """Minimize ||A [u, v] + b|| over a centered rectangle."""

    a_matrix = np.asarray(a_matrix, dtype=np.float32)
    b_vec = np.asarray(b_vec, dtype=np.float32)
    candidates: list[tuple[float, float]] = []

    try:
        uv = np.linalg.lstsq(a_matrix, -b_vec, rcond=None)[0]
        candidates.append((_clamp(float(uv[0]), -u_limit, u_limit), _clamp(float(uv[1]), -v_limit, v_limit)))
    except np.linalg.LinAlgError:
        candidates.append((0.0, 0.0))

    col_u = a_matrix[:, 0]
    col_v = a_matrix[:, 1]
    denom_u = float(np.dot(col_u, col_u))
    denom_v = float(np.dot(col_v, col_v))

    for fixed_u in (-u_limit, u_limit):
        if denom_v > 1e-12:
            v = -float(np.dot(col_v, b_vec + col_u * fixed_u)) / denom_v
        else:
            v = 0.0
        candidates.append((float(fixed_u), _clamp(v, -v_limit, v_limit)))

    for fixed_v in (-v_limit, v_limit):
        if denom_u > 1e-12:
            u = -float(np.dot(col_u, b_vec + col_v * fixed_v)) / denom_u
        else:
            u = 0.0
        candidates.append((_clamp(u, -u_limit, u_limit), float(fixed_v)))

    for fixed_u in (-u_limit, u_limit):
        for fixed_v in (-v_limit, v_limit):
            candidates.append((float(fixed_u), float(fixed_v)))

    best_distance = float("inf")
    best_u = 0.0
    best_v = 0.0
    for u, v in candidates:
        residual = a_matrix @ np.asarray([u, v], dtype=np.float32) + b_vec
        distance = float(np.linalg.norm(residual))
        if distance < best_distance:
            best_distance = distance
            best_u = float(u)
            best_v = float(v)
    return best_distance, best_u, best_v


def foot_toe_box_specs(
    positions: np.ndarray,
    rotations: np.ndarray,
    foot_index: int,
    toe_index: int,
    config: FootContactConfig = DEFAULT_CONFIG,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]]:
    """Return foot and toe boxes using the same basis choices as the viewer."""

    foot = np.asarray(positions[foot_index], dtype=np.float32)
    toe_pos = np.asarray(positions[toe_index], dtype=np.float32)
    toe_vector = toe_pos - foot

    up = np.asarray(rotations[foot_index, 0], dtype=np.float32).copy()
    forward = np.asarray(rotations[foot_index, 1], dtype=np.float32).copy()
    side = np.asarray(rotations[foot_index, 2], dtype=np.float32).copy()
    if float(np.dot(forward, toe_vector)) < 0.0:
        forward *= -1.0
    if float(up[1]) < 0.0:
        up *= -1.0
    forward = _normalize_axis(forward)
    side = _normalize_axis(side)
    up = _normalize_axis(up)

    foot_dims = (config.foot_length, config.foot_width, config.foot_height)
    heel_back = toe_pos - forward * config.foot_length
    foot_center = heel_back + forward * (config.foot_length * 0.5) + up * config.sole_vertical_offset

    toe_forward = np.asarray(rotations[toe_index, 0], dtype=np.float32).copy()
    toe_up = np.asarray(rotations[toe_index, 1], dtype=np.float32).copy()
    toe_side = np.asarray(rotations[toe_index, 2], dtype=np.float32).copy()
    if float(np.dot(toe_forward, toe_vector)) < 0.0:
        toe_forward *= -1.0
    if float(toe_up[1]) < 0.0:
        toe_up *= -1.0
    toe_forward = _normalize_axis(toe_forward)
    toe_side = _normalize_axis(toe_side)
    toe_up = _normalize_axis(toe_up)

    toe_dims = (config.toe_length, config.toe_width, config.toe_height)
    toe_center = toe_pos + toe_forward * (config.toe_length * 0.5) + toe_up * config.sole_vertical_offset

    return [
        ("foot", foot_center.astype(np.float32), forward, side, up, foot_dims),
        ("toe", toe_center.astype(np.float32), toe_forward, toe_side, toe_up, toe_dims),
    ]


def sole_rect_slide_distance(
    prev_spec: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]],
    cur_spec: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return minimum horizontal travel for one persistent point on the sole rectangle."""

    _part0, prev_center, prev_forward, prev_side, prev_up, prev_dims = prev_spec
    _part1, cur_center, cur_forward, cur_side, cur_up, cur_dims = cur_spec
    prev_dims_arr = np.asarray(prev_dims, dtype=np.float32)
    cur_dims_arr = np.asarray(cur_dims, dtype=np.float32)
    dims = np.minimum(prev_dims_arr, cur_dims_arr)

    prev_sole_center = prev_center - prev_up * (prev_dims_arr[2] * 0.5)
    cur_sole_center = cur_center - cur_up * (cur_dims_arr[2] * 0.5)
    a_matrix = np.stack(
        (
            _project_xz(cur_forward - prev_forward),
            _project_xz(cur_side - prev_side),
        ),
        axis=1,
    )
    b_vec = _project_xz(cur_sole_center - prev_sole_center)
    distance, u, v = _rect_least_squares_minimum(a_matrix, b_vec, dims[0] * 0.5, dims[1] * 0.5)
    prev_point = prev_sole_center + prev_forward * u + prev_side * v
    cur_point = cur_sole_center + cur_forward * u + cur_side * v
    return distance, prev_point.astype(np.float32), cur_point.astype(np.float32)


def foot_sole_slide_distance(
    prev_positions: np.ndarray,
    prev_rotations: np.ndarray,
    cur_positions: np.ndarray,
    cur_rotations: np.ndarray,
    foot_index: int,
    toe_index: int,
    config: FootContactConfig = DEFAULT_CONFIG,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """Return the smaller exact 2D sole slide from foot box or toe box."""

    prev_specs = foot_toe_box_specs(prev_positions, prev_rotations, foot_index, toe_index, config)
    cur_specs = foot_toe_box_specs(cur_positions, cur_rotations, foot_index, toe_index, config)
    best_distance = float("inf")
    best_prev_point = np.zeros(3, dtype=np.float32)
    best_cur_point = np.zeros(3, dtype=np.float32)
    best_part = ""
    for prev_spec, cur_spec in zip(prev_specs, cur_specs):
        distance, prev_point, cur_point = sole_rect_slide_distance(prev_spec, cur_spec)
        if distance < best_distance:
            best_distance = distance
            best_prev_point = prev_point
            best_cur_point = cur_point
            best_part = prev_spec[0]
    return best_distance, best_prev_point, best_cur_point, best_part


def foot_lowest_ground_height(
    positions: np.ndarray,
    rotations: np.ndarray,
    foot_index: int,
    toe_index: int,
    config: FootContactConfig = DEFAULT_CONFIG,
) -> tuple[float, np.ndarray, str]:
    """Return the lowest signed height from either the foot or toe collider."""

    best_height = float("inf")
    best_point = np.zeros(3, dtype=np.float32)
    best_part = ""
    for part, center, axis_x, axis_y, axis_z, dims in foot_toe_box_specs(
        positions, rotations, foot_index, toe_index, config
    ):
        point, height = box_lowest_point_signed_height(center, axis_x, axis_y, axis_z, dims, config.ground_y)
        if height < best_height:
            best_height = height
            best_point = point
            best_part = part
    return best_height, best_point, best_part


def foot_ground_height(
    positions: np.ndarray,
    rotations: np.ndarray,
    foot_index: int,
    toe_index: int,
    config: FootContactConfig = DEFAULT_CONFIG,
) -> tuple[float, np.ndarray, str]:
    return foot_lowest_ground_height(positions, rotations, foot_index, toe_index, config)


def is_foot_contact(
    height_m: np.ndarray | float,
    horizontal_speed_mps: np.ndarray | float,
    config: FootContactConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    height = np.asarray(height_m, dtype=np.float32)
    speed = np.asarray(horizontal_speed_mps, dtype=np.float32)
    return np.logical_and(
        height <= np.float32(config.height_threshold_m),
        speed <= np.float32(config.horizontal_speed_threshold_mps),
    )


def compute_contacts_from_arrays(
    global_joint_pos: np.ndarray,
    global_matrix: np.ndarray,
    bone_names: list[str],
    fps: float,
    up_axis: int,
    config: FootContactConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray]:
    positions = canonicalize_positions(global_joint_pos, up_axis) * np.float32(config.position_unit_scale)
    rotations = canonicalize_rotations(global_matrix[:, :, :3, :3], up_axis)
    name_to_index = {name: i for i, name in enumerate(bone_names)}
    sides = (("contactL", "foot_l", "ball_l"), ("contactR", "foot_r", "ball_r"))

    frame_count = int(positions.shape[0])
    contact_height = np.zeros((frame_count, 2), dtype=np.float32)
    contact_speed = np.zeros((frame_count, 2), dtype=np.float32)
    contact_point = np.zeros((frame_count, 2, 3), dtype=np.float32)
    contact_part = np.empty((frame_count, 2), dtype="<U4")
    contact_slide_distance = np.zeros((frame_count, 2), dtype=np.float32)
    contact_slide_point_prev = np.zeros((frame_count, 2, 3), dtype=np.float32)
    contact_slide_point_cur = np.zeros((frame_count, 2, 3), dtype=np.float32)
    contact_slide_part = np.empty((frame_count, 2), dtype="<U4")

    for side_i, (_contact_name, foot_name, toe_name) in enumerate(sides):
        if foot_name not in name_to_index or toe_name not in name_to_index:
            raise ValueError(f"Missing {foot_name}/{toe_name} bones for contact generation.")
        foot_index = name_to_index[foot_name]
        toe_index = name_to_index[toe_name]
        for frame in range(frame_count):
            height, point, part = foot_lowest_ground_height(
                positions[frame], rotations[frame], foot_index, toe_index, config
            )
            contact_height[frame, side_i] = height
            contact_point[frame, side_i] = point
            contact_part[frame, side_i] = part
            if frame > 0:
                slide_distance, prev_point, cur_point, slide_part = foot_sole_slide_distance(
                    positions[frame - 1],
                    rotations[frame - 1],
                    positions[frame],
                    rotations[frame],
                    foot_index,
                    toe_index,
                    config,
                )
                contact_slide_distance[frame, side_i] = slide_distance
                contact_slide_point_prev[frame, side_i] = prev_point
                contact_slide_point_cur[frame, side_i] = cur_point
                contact_slide_part[frame, side_i] = slide_part

    if frame_count > 1:
        contact_slide_distance[0] = contact_slide_distance[1]
        contact_slide_point_prev[0] = contact_slide_point_prev[1]
        contact_slide_point_cur[0] = contact_slide_point_cur[1]
        contact_slide_part[0] = contact_slide_part[1]
        contact_speed = contact_slide_distance * np.float32(fps)

    contacts = is_foot_contact(contact_height, contact_speed, config).astype(np.bool_)
    return {
        "contact_names": CONTACT_NAMES.copy(),
        "contacts": contacts,
        "contact_height_m": contact_height,
        "contact_speed_mps": contact_speed,
        "contact_lowest_point_m": contact_point,
        "contact_lowest_part": contact_part,
        "contact_slide_distance_m": contact_slide_distance,
        "contact_slide_point_prev_m": contact_slide_point_prev,
        "contact_slide_point_cur_m": contact_slide_point_cur,
        "contact_slide_part": contact_slide_part,
        "contact_height_threshold_m": np.asarray(config.height_threshold_m, dtype=np.float32),
        "contact_speed_threshold_mps": np.asarray(config.horizontal_speed_threshold_mps, dtype=np.float32),
    }
