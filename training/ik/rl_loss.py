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
    # The controller output is already root-local; this term measures pelvis
    # distance from the root in root-local horizontal XZ.
    pelvis_pos = predicted_output[:, :3]
    horizontal = torch.linalg.vector_norm(torch.stack((pelvis_pos[:, 0], pelvis_pos[:, 2]), dim=-1), dim=-1)
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


def compute_rl_loss(
    predicted_output: torch.Tensor,
    current_output: torch.Tensor,
    row_weight: torch.Tensor,
    active: torch.Tensor,
    cfg: RLLossConfig,
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

    return RLLossResult(
        total=sum(terms.values(), zero),
        terms=terms,
    )
