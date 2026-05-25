"""Quick smoke test: verify the new foot-pin / no-hover / foot-floor RL terms
compute, log under their expected names, and produce real gradients.

Runs from the project root (no editing of canonical trainer required).
"""

from __future__ import annotations

import torch

try:
    from . import ik_core as tl
    from .rl_loss import RLLossConfig, compute_rl_loss, foot_world_positions
except ImportError:
    import ik_core as tl
    from rl_loss import RLLossConfig, compute_rl_loss, foot_world_positions


def make_dummy_output(batch: int, device: torch.device) -> torch.Tensor:
    Jcore = 12  # Will be overridden when we infer from store; this is a self-contained probe.
    payload = tl.IK_PAYLOAD_DIM
    dim = 3 + 6 + Jcore * 6 + payload
    vec = torch.zeros(batch, dim, device=device)
    vec[:, 3] = 1.0
    vec[:, 7] = 1.0
    return vec, Jcore


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    batch = 8
    Jcore = 17  # matches the simple stepper2 skeleton: 25 bones - 4 EE - 4 ee-parents = doesn't matter here
    payload = tl.IK_PAYLOAD_DIM
    dim = 3 + 6 + Jcore * 6 + payload

    cur = torch.randn(batch, dim, device=device, requires_grad=False) * 0.01
    cur[:, 3] = 1.0
    cur[:, 7] = 1.0

    pred = cur.clone().detach().requires_grad_(True)

    leg_specs = tuple(s for s in tl.IK_PAYLOAD_SLICES if str(s["kind"]) == "leg")
    foot_left_pos_slice = leg_specs[0]["pos"]
    foot_right_pos_slice = leg_specs[1]["pos"]
    payload_start = 3 + 6 + Jcore * 6
    with torch.no_grad():
        cur[:, payload_start + foot_left_pos_slice.start : payload_start + foot_left_pos_slice.stop] = torch.tensor([0.1, -0.95, 0.0], device=device)
        cur[:, payload_start + foot_right_pos_slice.start : payload_start + foot_right_pos_slice.stop] = torch.tensor([-0.1, -0.95, 0.0], device=device)
        pred.copy_(cur)
        pred[:, payload_start + foot_left_pos_slice.start] += 0.05

    pred.requires_grad_(True)

    root_pos = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(batch, 3).contiguous()
    root_rot = torch.eye(3, device=device).expand(batch, 3, 3).contiguous()

    foot_world_cur = foot_world_positions(cur, root_pos, root_rot)
    foot_world_pred = foot_world_positions(pred, root_pos, root_rot)
    print("cur feet world (y should be ~0.05):", foot_world_cur[0].tolist())
    print("pred feet world:", foot_world_pred[0].tolist())

    rl_cfg = RLLossConfig(
        foot_pin_weight=1.0,
        foot_pin_height_threshold_m=0.08,
        foot_pin_height_temp_m=0.02,
        no_hover_weight=1.0,
        no_hover_height_threshold_m=0.10,
        no_hover_height_temp_m=0.03,
        foot_floor_weight=1.0,
        foot_floor_y_m=0.0,
        fps=30.0,
    )

    row_weight = torch.full((batch,), 1.0 / batch, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    res = compute_rl_loss(pred, cur, row_weight, active, rl_cfg, root_pos=root_pos, root_rot=root_rot)

    print("\nterms:")
    for k, v in res.terms.items():
        print(f"  {k}: {float(v.detach().cpu()):.6g}")
    print(f"total: {float(res.total.detach().cpu()):.6g}")

    res.total.backward()
    print(f"pred.grad norm: {float(pred.grad.norm().detach().cpu()):.6g}")
    pin_grad = pred.grad[:, payload_start + foot_left_pos_slice.start].abs().mean().item()
    print(f"|grad| on left foot x (should be > 0 because of pin loss): {pin_grad:.6g}")

    assert float(res.terms["rl_foot_pin"]) > 0, "foot pin loss should be positive when foot at ground slides"
    assert float(res.terms["rl_no_hover"]) >= 0, "no-hover loss should be non-negative"
    assert pin_grad > 0, "expected non-zero gradient on the sliding foot's x payload"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
