from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContactGeometryConfig:
    foot_length: float = 0.150
    foot_width: float = 0.110
    foot_height: float = 0.051
    toe_length: float = 0.065
    toe_width: float = 0.110
    toe_height: float = 0.050
    sole_vertical_offset: float = -0.006
    ground_y: float = 0.0
    height_threshold_m: float = 0.025
    speed_threshold_mps: float = 0.350
    gravity_mps2: float = 9.81


DEFAULT_GEOMETRY = ContactGeometryConfig()


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def basis_axes_from_direction(
    basis: torch.Tensor,
    direction: torch.Tensor,
    fallback_forward_axis: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    basis = normalize(basis)
    direction_n = normalize(direction)
    dots = (basis * direction_n.unsqueeze(-2)).sum(dim=-1)
    forward_index = torch.argmax(torch.abs(dots), dim=-1)
    if fallback_forward_axis >= 0:
        direction_len = torch.linalg.norm(direction, dim=-1)
        fallback = torch.full_like(forward_index, int(fallback_forward_axis))
        forward_index = torch.where(direction_len > 1e-8, forward_index, fallback)

    gather_index = forward_index[:, None, None].expand(-1, 1, 3)
    forward = basis.gather(1, gather_index).squeeze(1)
    forward_dot = dots.gather(1, forward_index[:, None])
    forward = torch.where(forward_dot < 0.0, -forward, forward)

    up_score = torch.abs(basis[:, :, 1])
    up_score = up_score.masked_fill(
        torch.arange(3, device=basis.device).unsqueeze(0) == forward_index.unsqueeze(1),
        -1.0,
    )
    up_index = torch.argmax(up_score, dim=-1)
    up = basis.gather(1, up_index[:, None, None].expand(-1, 1, 3)).squeeze(1)
    up = torch.where(up[:, 1:2] < 0.0, -up, up)

    side = normalize(torch.cross(up, forward, dim=-1))
    return normalize(forward), side, normalize(up)


def box_lowest_point_signed_height(
    center: torch.Tensor,
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    axis_z: torch.Tensor,
    dims: tuple[float, float, float],
    ground_y: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    axes = torch.stack((normalize(axis_x), normalize(axis_y), normalize(axis_z)), dim=-2)
    half = torch.tensor(dims, dtype=center.dtype, device=center.device) * 0.5
    signs = torch.where(axes[..., :, 1] > 0.0, -1.0, 1.0).to(center.dtype)
    point = center + (axes * (signs * half).unsqueeze(-1)).sum(dim=-2)
    return point, point[..., 1] - float(ground_y)


def foot_toe_box_specs(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    foot_index: int,
    toe_index: int,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]]]:
    foot = positions[:, foot_index]
    toe = positions[:, toe_index]
    toe_vec = toe - foot

    forward, side, up = basis_axes_from_direction(rotations[:, foot_index], toe_vec, 1)
    foot_dims = (cfg.foot_length, cfg.foot_width, cfg.foot_height)
    heel_back = toe - forward * cfg.foot_length
    foot_center = heel_back + forward * (cfg.foot_length * 0.5) + up * cfg.sole_vertical_offset

    toe_forward, toe_side, toe_up = basis_axes_from_direction(rotations[:, toe_index], toe_vec, 0)
    toe_dims = (cfg.toe_length, cfg.toe_width, cfg.toe_height)
    toe_center = toe + toe_forward * (cfg.toe_length * 0.5) + toe_up * cfg.sole_vertical_offset
    return [
        ("foot", foot_center, forward, side, up, foot_dims),
        ("toe", toe_center, toe_forward, toe_side, toe_up, toe_dims),
    ]


def foot_lowest_heights_and_points(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    foot_indices: tuple[int, int],
    toe_indices: tuple[int, int],
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lowest vertical foot/toe collider point for height checks.

    This is intentionally separate from sliding. Sliding is measured by
    `foot_slide_speeds`, which solves for the best 2D sole point over the whole
    foot/toe rectangle instead of reusing this lowest point.
    """
    heights = []
    points = []
    for foot_index, toe_index in zip(foot_indices, toe_indices):
        part_heights = []
        part_points = []
        for _name, center, axis_x, axis_y, axis_z, dims in foot_toe_box_specs(
            positions, rotations, foot_index, toe_index, cfg
        ):
            point, height = box_lowest_point_signed_height(center, axis_x, axis_y, axis_z, dims, cfg.ground_y)
            part_heights.append(height)
            part_points.append(point)
        h = torch.stack(part_heights, dim=-1)
        p = torch.stack(part_points, dim=-2)
        min_i = h.argmin(dim=-1)
        heights.append(h.gather(-1, min_i[:, None]).squeeze(-1))
        points.append(p.gather(1, min_i[:, None, None].expand(-1, 1, 3)).squeeze(1))
    return torch.stack(heights, dim=-1), torch.stack(points, dim=1)


def _rect_min_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    u_limit: float,
    v_limit: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ata = a.transpose(-1, -2) @ a
    eye = torch.eye(2, dtype=a.dtype, device=a.device).expand_as(ata)
    rhs = -(a.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
    uv = torch.linalg.solve(ata + eye * 1e-6, rhs)
    uv = torch.stack((uv[:, 0].clamp(-u_limit, u_limit), uv[:, 1].clamp(-v_limit, v_limit)), dim=-1)
    candidates = [uv]
    col_u = a[:, :, 0]
    col_v = a[:, :, 1]
    denom_u = (col_u * col_u).sum(dim=-1).clamp_min(1e-8)
    denom_v = (col_v * col_v).sum(dim=-1).clamp_min(1e-8)
    for fixed_u in (-u_limit, u_limit):
        fu = torch.full_like(denom_v, float(fixed_u))
        v = -((col_v * (b + col_u * fu[:, None])).sum(dim=-1) / denom_v).clamp(-v_limit, v_limit)
        candidates.append(torch.stack((fu, v), dim=-1))
    for fixed_v in (-v_limit, v_limit):
        fv = torch.full_like(denom_u, float(fixed_v))
        u = -((col_u * (b + col_v * fv[:, None])).sum(dim=-1) / denom_u).clamp(-u_limit, u_limit)
        candidates.append(torch.stack((u, fv), dim=-1))
    for fixed_u in (-u_limit, u_limit):
        for fixed_v in (-v_limit, v_limit):
            candidates.append(torch.stack((torch.full_like(denom_u, fixed_u), torch.full_like(denom_u, fixed_v)), dim=-1))
    cand = torch.stack(candidates, dim=1)
    residual = torch.einsum("bij,bkj->bki", a, cand) + b[:, None, :]
    dist = torch.linalg.norm(residual, dim=-1)
    best_i = dist.argmin(dim=-1)
    best_uv = cand.gather(1, best_i[:, None, None].expand(-1, 1, 2)).squeeze(1)
    best_dist = dist.gather(1, best_i[:, None]).squeeze(1)
    return best_dist, best_uv[:, 0], best_uv[:, 1]


def sole_rect_slide_distance(
    prev_spec: tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]],
    cur_spec: tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]],
) -> torch.Tensor:
    """Return the best horizontal slide of any point on the sole rectangle."""
    _n0, prev_center, prev_forward, prev_side, prev_up, prev_dims = prev_spec
    _n1, cur_center, cur_forward, cur_side, cur_up, cur_dims = cur_spec
    dims = tuple(min(a, b) for a, b in zip(prev_dims, cur_dims))
    prev_sole = prev_center - prev_up * (prev_dims[2] * 0.5)
    cur_sole = cur_center - cur_up * (cur_dims[2] * 0.5)
    a = torch.stack(
        (
            (cur_forward - prev_forward)[:, [0, 2]],
            (cur_side - prev_side)[:, [0, 2]],
        ),
        dim=-1,
    )
    b = (cur_sole - prev_sole)[:, [0, 2]]
    dist, _u, _v = _rect_min_distance(a, b, dims[0] * 0.5, dims[1] * 0.5)
    return dist


def sole_rect_contact_point_distance(
    prev_spec: tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]],
    cur_spec: tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[float, float, float]],
) -> torch.Tensor:
    """Return the best full 3D velocity distance of any point on the sole rectangle."""
    _n0, prev_center, prev_forward, prev_side, prev_up, prev_dims = prev_spec
    _n1, cur_center, cur_forward, cur_side, cur_up, cur_dims = cur_spec
    dims = tuple(min(a, b) for a, b in zip(prev_dims, cur_dims))
    prev_sole = prev_center - prev_up * (prev_dims[2] * 0.5)
    cur_sole = cur_center - cur_up * (cur_dims[2] * 0.5)
    a = torch.stack((cur_forward - prev_forward, cur_side - prev_side), dim=-1)
    b = cur_sole - prev_sole
    dist, _u, _v = _rect_min_distance(a, b, dims[0] * 0.5, dims[1] * 0.5)
    return dist


def foot_slide_speeds(
    prev_positions: torch.Tensor,
    prev_rotations: torch.Tensor,
    cur_positions: torch.Tensor,
    cur_rotations: torch.Tensor,
    foot_indices: tuple[int, int],
    toe_indices: tuple[int, int],
    fps: float,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> torch.Tensor:
    """Return exact 2D sole slide speed for each foot.

    For each side this evaluates both the foot and toe sole rectangles, solves
    the tiny constrained least-squares problem in ground-plane XZ for each, and
    returns the smaller speed. Lowest-point height is not used here.
    """
    speeds = []
    for foot_index, toe_index in zip(foot_indices, toe_indices):
        prev_specs = foot_toe_box_specs(prev_positions, prev_rotations, foot_index, toe_index, cfg)
        cur_specs = foot_toe_box_specs(cur_positions, cur_rotations, foot_index, toe_index, cfg)
        distances = [sole_rect_slide_distance(a, b) for a, b in zip(prev_specs, cur_specs)]
        speeds.append(torch.stack(distances, dim=-1).amin(dim=-1) * float(fps))
    return torch.stack(speeds, dim=-1)


def foot_contact_point_speeds(
    prev_positions: torch.Tensor,
    prev_rotations: torch.Tensor,
    cur_positions: torch.Tensor,
    cur_rotations: torch.Tensor,
    foot_indices: tuple[int, int],
    toe_indices: tuple[int, int],
    fps: float,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> torch.Tensor:
    """Return the minimum full 3D speed of any persistent foot/toe sole point."""
    speeds = []
    for foot_index, toe_index in zip(foot_indices, toe_indices):
        prev_specs = foot_toe_box_specs(prev_positions, prev_rotations, foot_index, toe_index, cfg)
        cur_specs = foot_toe_box_specs(cur_positions, cur_rotations, foot_index, toe_index, cfg)
        distances = [sole_rect_contact_point_distance(a, b) for a, b in zip(prev_specs, cur_specs)]
        speeds.append(torch.stack(distances, dim=-1).amin(dim=-1) * float(fps))
    return torch.stack(speeds, dim=-1)


def contact_logits_loss(logits: torch.Tensor, target_contacts: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target_contacts.float())


def penetration_loss(heights: torch.Tensor, cfg: ContactGeometryConfig = DEFAULT_GEOMETRY) -> torch.Tensor:
    return F.relu(-heights).square().mean()


def contact_height_loss(
    heights: torch.Tensor,
    contact_prob: torch.Tensor,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> torch.Tensor:
    return (contact_prob * F.relu(heights - cfg.height_threshold_m).square()).mean()


def sliding_loss(
    speeds: torch.Tensor,
    contact_prob: torch.Tensor,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> torch.Tensor:
    return (contact_prob * F.relu(speeds - cfg.speed_threshold_mps).square()).mean()


def center_of_mass(positions: torch.Tensor, mass_weights: torch.Tensor) -> torch.Tensor:
    weights = mass_weights.to(device=positions.device, dtype=positions.dtype)
    weights = weights / weights.sum().clamp_min(1e-8)
    return (positions * weights[None, :, None]).sum(dim=1)


def freefall_loss(
    prev_com: torch.Tensor,
    cur_com: torch.Tensor,
    next_com: torch.Tensor,
    no_contact_prob: torch.Tensor,
    fps: float,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = 1.0 / float(fps)
    velocity = (cur_com - prev_com) / dt
    expected = cur_com + velocity * dt
    expected_y = expected[:, 1] - 0.5 * cfg.gravity_mps2 * dt * dt
    error_y = next_com[:, 1] - expected_y
    denom = torch.clamp(torch.abs(expected_y - cur_com[:, 1]), min=0.01)
    relative_error = torch.abs(error_y) / denom
    return (no_contact_prob * error_y.square()).mean(), relative_error


def termination_mask(
    heights: torch.Tensor,
    speeds: torch.Tensor,
    contact_prob: torch.Tensor,
    freefall_relative_error: torch.Tensor,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
    include_freefall: bool = True,
) -> torch.Tensor:
    contact_on = contact_prob > 0.5
    penetration_bad = heights < -cfg.height_threshold_m
    high_bad = torch.logical_and(contact_on, heights > 1.5 * cfg.height_threshold_m)
    slide_bad = torch.logical_and(contact_on, speeds > 1.5 * cfg.speed_threshold_mps)
    if include_freefall:
        freefall_bad = torch.logical_and((contact_prob < 0.5).all(dim=-1), freefall_relative_error > 0.10)
    else:
        freefall_bad = torch.zeros_like(freefall_relative_error, dtype=torch.bool)
    return torch.logical_or(torch.logical_or(penetration_bad.any(dim=-1), high_bad.any(dim=-1)), torch.logical_or(slide_bad.any(dim=-1), freefall_bad))


def termination_severity(
    heights: torch.Tensor,
    speeds: torch.Tensor,
    contact_prob: torch.Tensor,
    freefall_relative_error: torch.Tensor,
    cfg: ContactGeometryConfig = DEFAULT_GEOMETRY,
    include_freefall: bool = True,
) -> torch.Tensor:
    height_eps = max(cfg.height_threshold_m, 1e-6)
    speed_limit = max(1.5 * cfg.speed_threshold_mps, 1e-6)
    high_limit = max(1.5 * cfg.height_threshold_m, 1e-6)
    penetration = F.relu((-heights - cfg.height_threshold_m) / height_eps)
    high_contact = contact_prob * F.relu((heights - high_limit) / high_limit)
    slide = contact_prob * F.relu((speeds - speed_limit) / speed_limit)
    if include_freefall:
        no_contact_prob = (1.0 - contact_prob[:, 0]) * (1.0 - contact_prob[:, 1])
        freefall = no_contact_prob * F.relu((freefall_relative_error - 0.10) / 0.10)
    else:
        freefall = torch.zeros_like(freefall_relative_error)
    return penetration.sum(dim=-1) + high_contact.sum(dim=-1) + slide.sum(dim=-1) + freefall
