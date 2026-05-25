from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

try:
    from . import ik_core as tl
except ImportError:
    import ik_core as tl


RL_TERM_NAMES = (
    "rl_pelvis_root_horizontal",
    "rl_pelvis_root_rotation",
    "rl_end_effector_location",
    "rl_end_effector_rotation",
    "rl_end_effector_velocity",
    "rl_end_effector_angular_velocity",
    "rl_pelvis_velocity",
    "rl_pelvis_angular_velocity",
    "rl_core_angular_velocity",
    "rl_foot_pin",
    "rl_no_hover",
    "rl_foot_floor",
    "rl_foot_ceiling",
    "rl_pelvis_height",
)

_LIMIT_TENSOR_CACHE: dict[tuple[tuple[float, ...], str, torch.dtype], torch.Tensor] = {}


@dataclass(frozen=True)
class RLLossConfig:
    pelvis_root_horizontal_weight: float = 0.0
    pelvis_root_horizontal_limit_m: float = 1.03015552
    pelvis_root_rotation_weight: float = 0.0
    pelvis_root_rotation_limit_deg: float = 156.715400
    end_effector_location_weight: float = 0.0
    end_effector_location_limit_m: tuple[float, float, float, float] = (
        1.28302503,
        1.19828948,
        0.70878882,
        0.76501006,
    )
    end_effector_rotation_weight: float = 0.0
    end_effector_rotation_limit_deg: tuple[float, float, float, float] = (
        188.132536,
        188.136525,
        155.362205,
        188.914364,
    )
    end_effector_velocity_weight: float = 0.0
    end_effector_velocity_limit_mps: float = 5.35648992
    end_effector_angular_velocity_weight: float = 0.0
    end_effector_angular_velocity_limit_deg_s: float = 1259.712634
    pelvis_velocity_weight: float = 0.0
    pelvis_velocity_limit_mps: float = 1.34163226
    pelvis_angular_velocity_weight: float = 0.0
    pelvis_angular_velocity_limit_deg_s: float = 550.908289
    core_angular_velocity_weight: float = 0.0
    core_angular_velocity_limit_deg_s: float = 315.824318
    # Foot-pinning / no-hover terms operate on world-space foot positions and use a
    # purely geometric soft-contact mask (sigmoid of foot height above ground).
    # No dataset contact labels are used; this is consistent with the IK journal
    # rule "No contact labels anywhere. Foot losses are geometry/envelope based."
    foot_pin_weight: float = 0.0
    foot_pin_height_threshold_m: float = 0.08
    foot_pin_height_temp_m: float = 0.025
    foot_pin_speed_floor_mps: float = 0.0
    no_hover_weight: float = 0.0
    no_hover_height_threshold_m: float = 0.10
    no_hover_height_temp_m: float = 0.03
    foot_floor_weight: float = 0.0
    foot_floor_y_m: float = 0.0
    # Foot-ceiling term: penalises a foot whose world-y exceeds a ceiling.
    # Combined with no_hover and foot_pin this prevents the "kangaroo skip"
    # local minimum in which one foot stays permanently in the air and the
    # other shuffles. The squared-excess shape keeps short, low swings cheap
    # while making sustained high carriage expensive.
    foot_ceiling_weight: float = 0.0
    foot_ceiling_y_m: float = 0.15
    # Pelvis-height-in-root-local term. Closes the "crouch" loophole where the
    # agent lowers the pelvis (pelvis_local.Z) so the planted foot's distance to
    # root stays within end_effector_location_limit_m even as it drifts backward
    # in local-Y. Penalises absolute deviation of pelvis_local.Z from the
    # target. (Recall: root-local Z is "up" in this IK convention.)
    pelvis_height_weight: float = 0.0
    pelvis_height_target_m: float = 0.886
    pelvis_height_tolerance_m: float = 0.05
    fps: float = 30.0

    @property
    def enabled(self) -> bool:
        return any(float(getattr(self, name)) != 0.0 for name in self._weight_fields())

    @staticmethod
    def _weight_fields() -> tuple[str, ...]:
        return (
            "pelvis_root_horizontal_weight",
            "pelvis_root_rotation_weight",
            "end_effector_location_weight",
            "end_effector_rotation_weight",
            "end_effector_velocity_weight",
            "end_effector_angular_velocity_weight",
            "pelvis_velocity_weight",
            "pelvis_angular_velocity_weight",
            "core_angular_velocity_weight",
            "foot_pin_weight",
            "no_hover_weight",
            "foot_floor_weight",
            "foot_ceiling_weight",
            "pelvis_height_weight",
        )

    @property
    def world_foot_enabled(self) -> bool:
        return (
            float(self.foot_pin_weight) != 0.0
            or float(self.no_hover_weight) != 0.0
            or float(self.foot_floor_weight) != 0.0
            or float(self.foot_ceiling_weight) != 0.0
        )

    def enabled_terms(self) -> tuple[str, ...]:
        terms: list[str] = []
        if float(self.pelvis_root_horizontal_weight) != 0.0:
            terms.append("rl_pelvis_root_horizontal")
        if float(self.pelvis_root_rotation_weight) != 0.0:
            terms.append("rl_pelvis_root_rotation")
        if float(self.end_effector_location_weight) != 0.0:
            terms.append("rl_end_effector_location")
        if float(self.end_effector_rotation_weight) != 0.0:
            terms.append("rl_end_effector_rotation")
        if float(self.end_effector_velocity_weight) != 0.0:
            terms.append("rl_end_effector_velocity")
        if float(self.end_effector_angular_velocity_weight) != 0.0:
            terms.append("rl_end_effector_angular_velocity")
        if float(self.pelvis_velocity_weight) != 0.0:
            terms.append("rl_pelvis_velocity")
        if float(self.pelvis_angular_velocity_weight) != 0.0:
            terms.append("rl_pelvis_angular_velocity")
        if float(self.core_angular_velocity_weight) != 0.0:
            terms.append("rl_core_angular_velocity")
        if float(self.foot_pin_weight) != 0.0:
            terms.append("rl_foot_pin")
        if float(self.no_hover_weight) != 0.0:
            terms.append("rl_no_hover")
        if float(self.foot_floor_weight) != 0.0:
            terms.append("rl_foot_floor")
        if float(self.foot_ceiling_weight) != 0.0:
            terms.append("rl_foot_ceiling")
        if float(self.pelvis_height_weight) != 0.0:
            terms.append("rl_pelvis_height")
        return tuple(terms)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class RLLossResult:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]


def zero_result(device: torch.device, dtype: torch.dtype = torch.float32) -> RLLossResult:
    zero = torch.zeros((), dtype=dtype, device=device)
    return RLLossResult(total=zero, terms={name: zero for name in RL_TERM_NAMES})


def pelvis_root_horizontal_excess_rows(
    predicted_output: torch.Tensor,
    limit_m: float,
) -> torch.Tensor:
    # Root-local convention in this IK pipeline: local +Z is "up" (vertical),
    # local -Y is character forward (see IK_CHARACTER_FORWARD). The horizontal
    # plane is therefore (X, Y), not (X, Z). Using (X, Z) silently lets the
    # pelvis drift unbounded along local-Y when the root translates forward,
    # which is exactly the failure mode this term is meant to prevent.
    pelvis_pos = predicted_output[:, :3]
    horizontal = torch.linalg.vector_norm(pelvis_pos[:, :2], dim=-1)
    return torch.relu(horizontal - float(limit_m))


def pelvis_root_rotation_excess_rows(
    predicted_output: torch.Tensor,
    limit_deg: float,
) -> torch.Tensor:
    pelvis_rot = tl.rotation_6d_to_matrix(predicted_output[:, 3:9])
    identity = torch.eye(3, dtype=pelvis_rot.dtype, device=pelvis_rot.device).expand_as(pelvis_rot)
    angle = tl.geodesic_angles(pelvis_rot, identity)
    limit_rad = float(limit_deg) * torch.pi / 180.0
    return torch.relu(angle - limit_rad)


def _payload_start(output: torch.Tensor) -> int:
    return int(output.shape[-1]) - int(tl.IK_PAYLOAD_DIM)


def _limit_tensor(
    limit: float | tuple[float, ...] | list[float],
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(limit, (tuple, list)):
        if len(limit) != count:
            raise ValueError(f"Expected {count} limits, got {len(limit)}")
        values = tuple(float(v) for v in limit)
        key = (values, str(device), dtype)
        cached = _LIMIT_TENSOR_CACHE.get(key)
        if cached is None:
            cached = torch.tensor(values, dtype=dtype, device=device)
            _LIMIT_TENSOR_CACHE[key] = cached
        return cached
    return torch.full((count,), float(limit), dtype=dtype, device=device)


def _payload_stack(output: torch.Tensor, part: str) -> torch.Tensor:
    payload_start = _payload_start(output)
    parts: list[torch.Tensor] = []
    for spec in tl.IK_PAYLOAD_SLICES:
        sl = spec[part]
        assert isinstance(sl, slice)
        start = payload_start + int(sl.start)
        stop = payload_start + int(sl.stop)
        parts.append(output[:, start:stop])
    return torch.stack(parts, dim=1)


def end_effector_location_excess_rows(
    predicted_output: torch.Tensor,
    limit_m: float | tuple[float, ...] | list[float],
) -> torch.Tensor:
    pos = _payload_stack(predicted_output, "pos")
    limits = _limit_tensor(limit_m, pos.shape[1], pos.device, pos.dtype)
    dist = torch.linalg.vector_norm(pos, dim=-1)
    return torch.relu(dist - limits[None, :])


def end_effector_rotation_excess_rows(
    predicted_output: torch.Tensor,
    limit_deg: float | tuple[float, ...] | list[float],
) -> torch.Tensor:
    rot6 = _payload_stack(predicted_output, "rot6")
    rot = tl.rotation_6d_to_matrix(rot6.reshape(-1, 6))
    identity = torch.eye(3, dtype=rot.dtype, device=rot.device).expand_as(rot)
    angle = tl.geodesic_angles(rot, identity).reshape(rot6.shape[:-1])
    limits_deg = _limit_tensor(limit_deg, rot6.shape[1], rot6.device, rot6.dtype)
    limits_rad = limits_deg * (torch.pi / 180.0)
    return torch.relu(angle - limits_rad[None, :])


def end_effector_velocity_excess_rows(
    current_output: torch.Tensor,
    predicted_output: torch.Tensor,
    limit_mps: float,
    fps: float,
) -> torch.Tensor:
    cur = _payload_stack(current_output, "pos")
    pred = _payload_stack(predicted_output, "pos")
    speed = torch.linalg.vector_norm(pred - cur, dim=-1) * float(fps)
    return torch.relu(speed - float(limit_mps))


def end_effector_angular_velocity_excess_rows(
    current_output: torch.Tensor,
    predicted_output: torch.Tensor,
    limit_deg_s: float,
    fps: float,
) -> torch.Tensor:
    cur = _payload_stack(current_output, "rot6")
    pred = _payload_stack(predicted_output, "rot6")
    return angular_velocity_excess_rows(cur, pred, limit_deg_s, fps)


def linear_velocity_excess_rows(
    current_pos: torch.Tensor,
    predicted_pos: torch.Tensor,
    limit_mps: float,
    fps: float,
) -> torch.Tensor:
    speed = torch.linalg.vector_norm(predicted_pos - current_pos, dim=-1) * float(fps)
    return torch.relu(speed - float(limit_mps))


def angular_velocity_excess_rows(
    current_rot6: torch.Tensor,
    predicted_rot6: torch.Tensor,
    limit_deg_s: float,
    fps: float,
) -> torch.Tensor:
    if current_rot6.numel() == 0:
        return current_rot6.new_zeros(current_rot6.shape[:-1])
    current_rot = tl.rotation_6d_to_matrix(current_rot6.reshape(-1, 6))
    predicted_rot = tl.rotation_6d_to_matrix(predicted_rot6.reshape(-1, 6))
    angle = tl.geodesic_angles(predicted_rot, current_rot).reshape(current_rot6.shape[:-1])
    limit_rad_s = float(limit_deg_s) * torch.pi / 180.0
    return torch.relu(angle * float(fps) - limit_rad_s)


def _leg_pos_slices() -> tuple[slice, slice]:
    leg_specs = tuple(spec for spec in tl.IK_PAYLOAD_SLICES if str(spec["kind"]) == "leg")
    assert len(leg_specs) == 2, f"expected 2 legs in IK_PAYLOAD_SLICES, got {len(leg_specs)}"
    left = leg_specs[0]["pos"]
    right = leg_specs[1]["pos"]
    assert isinstance(left, slice) and isinstance(right, slice)
    return left, right


def foot_world_positions(
    output_vec: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
) -> torch.Tensor:
    """Return foot world positions, shape (B, 2, 3) ordered (left, right).

    `output_vec` is an IK output vector expressed in the current root frame.
    Foot positions are read from the IK payload (root-local) and transformed
    to world via the supplied root transform: world = local @ root_rot + root_pos.
    """

    payload_start = _payload_start(output_vec)
    left_slice, right_slice = _leg_pos_slices()
    left_local = output_vec[:, payload_start + left_slice.start : payload_start + left_slice.stop]
    right_local = output_vec[:, payload_start + right_slice.start : payload_start + right_slice.stop]
    local = torch.stack((left_local, right_local), dim=1)
    world = torch.einsum("blk,bkj->blj", local, root_rot) + root_pos[:, None, :]
    return world


def _soft_contact(height_above_ground: torch.Tensor, threshold_m: float, temp_m: float) -> torch.Tensor:
    temp = max(float(temp_m), 1e-4)
    return torch.sigmoid((float(threshold_m) - height_above_ground) / temp)


def foot_pin_excess_rows(
    foot_world_cur: torch.Tensor,
    foot_world_pred: torch.Tensor,
    threshold_m: float,
    temp_m: float,
    speed_floor_mps: float,
    fps: float,
) -> torch.Tensor:
    """Per-row, per-foot squared horizontal world-velocity weighted by soft contact.

    Returns a tensor shaped (B, 2) holding `c * relu(speed_h - floor)^2` for the
    left and right foot. `c` is a sigmoid contact mask computed from world-y of
    `foot_world_cur` (i.e. the foot we *previously* planted is the one we forbid
    from sliding). `floor` allows ignoring tiny numerical drift below a floor.
    """

    height_cur = foot_world_cur[..., 1]
    contact = _soft_contact(height_cur, threshold_m, temp_m)
    horiz_delta = foot_world_pred[..., (0, 2)] - foot_world_cur[..., (0, 2)]
    speed_h = torch.linalg.vector_norm(horiz_delta, dim=-1) * float(fps)
    excess = torch.relu(speed_h - float(speed_floor_mps))
    return contact * excess.square()


def no_hover_excess_rows(
    foot_world: torch.Tensor,
    threshold_m: float,
    temp_m: float,
) -> torch.Tensor:
    """Per-row hover penalty: large when BOTH feet are above the contact band.

    Returns a tensor shaped (B,) holding `(1 - c_l) * (1 - c_r)`, computed
    from `foot_world` (B, 2, 3). Use the predicted next-frame foot world
    positions so the model is graded on its *committed* next pose.
    """

    height = foot_world[..., 1]
    c_l = _soft_contact(height[..., 0], threshold_m, temp_m)
    c_r = _soft_contact(height[..., 1], threshold_m, temp_m)
    return (1.0 - c_l) * (1.0 - c_r)


def foot_floor_excess_rows(
    foot_world: torch.Tensor,
    floor_y_m: float,
) -> torch.Tensor:
    """Per-row, per-foot squared distance the foot sinks below `floor_y_m`."""

    below = torch.relu(float(floor_y_m) - foot_world[..., 1])
    return below.square()


def foot_ceiling_excess_rows(
    foot_world: torch.Tensor,
    ceiling_y_m: float,
) -> torch.Tensor:
    """Per-row, per-foot squared distance the foot rises above `ceiling_y_m`.

    Acts as a soft "max swing height" cap. Without it the optimiser can park
    one foot permanently in the air and shuffle the other one, satisfying
    no_hover/foot_pin/EE_location while never walking. Penalising sustained
    high carriage forces both feet to spend most of the rollout near the
    ground, which in turn forces alternation when the root keeps translating.
    """

    above = torch.relu(foot_world[..., 1] - float(ceiling_y_m))
    return above.square()


def pelvis_height_excess_rows(
    predicted_output: torch.Tensor,
    target_m: float,
    tolerance_m: float,
) -> torch.Tensor:
    """Per-row deviation of pelvis-local Z (root-local up axis) from a target
    height, ignoring deviations inside the tolerance band. Returns shape (B,).

    This is the "stand at natural height" constraint. Combined with the foot
    end_effector_location limit, it forces the agent to lift and swing feet
    instead of crouching to keep a planted foot within reach.
    """

    pelvis_z = predicted_output[:, 2]
    deviation = (pelvis_z - float(target_m)).abs()
    return torch.relu(deviation - float(tolerance_m))


def compute_rl_loss(
    predicted_output: torch.Tensor,
    current_output: torch.Tensor,
    row_weight: torch.Tensor,
    active: torch.Tensor,
    cfg: RLLossConfig,
    root_pos: torch.Tensor | None = None,
    root_rot: torch.Tensor | None = None,
) -> RLLossResult:
    if not cfg.enabled:
        return zero_result(predicted_output.device, predicted_output.dtype)

    active_f = active.to(dtype=predicted_output.dtype)
    weights = row_weight.to(dtype=predicted_output.dtype) * active_f
    zero = torch.zeros((), dtype=predicted_output.dtype, device=predicted_output.device)
    terms = {name: zero for name in RL_TERM_NAMES}

    pelvis_excess = pelvis_root_horizontal_excess_rows(
        predicted_output,
        cfg.pelvis_root_horizontal_limit_m,
    )
    terms["rl_pelvis_root_horizontal"] = (
        (pelvis_excess.square() * weights).sum() * float(cfg.pelvis_root_horizontal_weight)
    )

    pelvis_rot_excess = pelvis_root_rotation_excess_rows(
        predicted_output,
        cfg.pelvis_root_rotation_limit_deg,
    )
    terms["rl_pelvis_root_rotation"] = (
        (pelvis_rot_excess.square() * weights).sum() * float(cfg.pelvis_root_rotation_weight)
    )

    ee_loc_excess = end_effector_location_excess_rows(
        predicted_output,
        cfg.end_effector_location_limit_m,
    )
    terms["rl_end_effector_location"] = (
        (ee_loc_excess.square().mean(dim=-1) * weights).sum() * float(cfg.end_effector_location_weight)
    )

    ee_rot_excess = end_effector_rotation_excess_rows(
        predicted_output,
        cfg.end_effector_rotation_limit_deg,
    )
    terms["rl_end_effector_rotation"] = (
        (ee_rot_excess.square().mean(dim=-1) * weights).sum() * float(cfg.end_effector_rotation_weight)
    )

    ee_excess = end_effector_velocity_excess_rows(
        current_output,
        predicted_output,
        cfg.end_effector_velocity_limit_mps,
        cfg.fps,
    )
    terms["rl_end_effector_velocity"] = (
        (ee_excess.square().mean(dim=-1) * weights).sum() * float(cfg.end_effector_velocity_weight)
    )

    ee_ang_excess = end_effector_angular_velocity_excess_rows(
        current_output,
        predicted_output,
        cfg.end_effector_angular_velocity_limit_deg_s,
        cfg.fps,
    )
    terms["rl_end_effector_angular_velocity"] = (
        (ee_ang_excess.square().mean(dim=-1) * weights).sum() * float(cfg.end_effector_angular_velocity_weight)
    )

    pelvis_vel_excess = linear_velocity_excess_rows(
        current_output[:, :3],
        predicted_output[:, :3],
        cfg.pelvis_velocity_limit_mps,
        cfg.fps,
    )
    terms["rl_pelvis_velocity"] = (
        (pelvis_vel_excess.square() * weights).sum() * float(cfg.pelvis_velocity_weight)
    )

    pelvis_ang_excess = angular_velocity_excess_rows(
        current_output[:, 3:9],
        predicted_output[:, 3:9],
        cfg.pelvis_angular_velocity_limit_deg_s,
        cfg.fps,
    )
    terms["rl_pelvis_angular_velocity"] = (
        (pelvis_ang_excess.square() * weights).sum() * float(cfg.pelvis_angular_velocity_weight)
    )

    core_start = 9
    core_stop = _payload_start(predicted_output)
    core_cur = current_output[:, core_start:core_stop].reshape(current_output.shape[0], -1, 6)
    core_pred = predicted_output[:, core_start:core_stop].reshape(predicted_output.shape[0], -1, 6)
    core_ang_excess = angular_velocity_excess_rows(
        core_cur,
        core_pred,
        cfg.core_angular_velocity_limit_deg_s,
        cfg.fps,
    )
    if core_ang_excess.numel() > 0:
        terms["rl_core_angular_velocity"] = (
            (core_ang_excess.square().mean(dim=-1) * weights).sum() * float(cfg.core_angular_velocity_weight)
        )

    if cfg.world_foot_enabled and root_pos is not None and root_rot is not None:
        foot_world_cur = foot_world_positions(current_output, root_pos, root_rot)
        foot_world_pred = foot_world_positions(predicted_output, root_pos, root_rot)
        if float(cfg.foot_pin_weight) != 0.0:
            pin_rows = foot_pin_excess_rows(
                foot_world_cur,
                foot_world_pred,
                cfg.foot_pin_height_threshold_m,
                cfg.foot_pin_height_temp_m,
                cfg.foot_pin_speed_floor_mps,
                cfg.fps,
            )
            terms["rl_foot_pin"] = (pin_rows.mean(dim=-1) * weights).sum() * float(cfg.foot_pin_weight)
        if float(cfg.no_hover_weight) != 0.0:
            hover_rows = no_hover_excess_rows(
                foot_world_pred,
                cfg.no_hover_height_threshold_m,
                cfg.no_hover_height_temp_m,
            )
            terms["rl_no_hover"] = (hover_rows * weights).sum() * float(cfg.no_hover_weight)
        if float(cfg.foot_floor_weight) != 0.0:
            floor_rows = foot_floor_excess_rows(foot_world_pred, cfg.foot_floor_y_m)
            terms["rl_foot_floor"] = (floor_rows.mean(dim=-1) * weights).sum() * float(cfg.foot_floor_weight)
        if float(cfg.foot_ceiling_weight) != 0.0:
            ceiling_rows = foot_ceiling_excess_rows(foot_world_pred, cfg.foot_ceiling_y_m)
            terms["rl_foot_ceiling"] = (
                (ceiling_rows.mean(dim=-1) * weights).sum() * float(cfg.foot_ceiling_weight)
            )

    if float(cfg.pelvis_height_weight) != 0.0:
        pelvis_h_rows = pelvis_height_excess_rows(
            predicted_output,
            cfg.pelvis_height_target_m,
            cfg.pelvis_height_tolerance_m,
        )
        terms["rl_pelvis_height"] = (
            (pelvis_h_rows.square() * weights).sum() * float(cfg.pelvis_height_weight)
        )

    return RLLossResult(
        total=sum(terms.values(), zero),
        terms=terms,
    )
