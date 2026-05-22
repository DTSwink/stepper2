from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentMassTable:
    # 50th percentile American male crewmember segment masses from NASA-STD-3000,
    # as mirrored in NASA/MSIS body segment mass data. NASA's current standards
    # page documents body-segment mass properties in Appendix E.5.
    # Values are kg and are normalized before COM use.
    head: float = 4.40
    neck: float = 1.10
    thorax: float = 26.11
    abdomen: float = 2.50
    pelvis: float = 12.30
    upper_arm: float = 2.00
    forearm: float = 1.45
    hand: float = 0.53
    hip_flap: float = 3.64
    thigh_minus_flap: float = 6.70
    calf: float = 4.04
    foot: float = 1.01


NASA_50TH_PERCENTILE_MALE = SegmentMassTable()


def bone_masses_for_names(bone_names: list[str], table: SegmentMassTable = NASA_50TH_PERCENTILE_MALE) -> list[float]:
    masses: dict[str, float] = {
        "root": 0.0,
        "pelvis": table.pelvis,
        "spine_01": (table.abdomen + table.thorax) / 5.0,
        "spine_02": (table.abdomen + table.thorax) / 5.0,
        "spine_03": (table.abdomen + table.thorax) / 5.0,
        "spine_04": (table.abdomen + table.thorax) / 5.0,
        "spine_05": (table.abdomen + table.thorax) / 5.0,
        "neck_01": table.neck * 0.5,
        "neck_02": table.neck * 0.5,
        "head": table.head,
        "clavicle_l": 0.0,
        "upperarm_l": table.upper_arm,
        "lowerarm_l": table.forearm,
        "hand_l": table.hand,
        "clavicle_r": 0.0,
        "upperarm_r": table.upper_arm,
        "lowerarm_r": table.forearm,
        "hand_r": table.hand,
        "thigh_l": table.hip_flap + table.thigh_minus_flap,
        "calf_l": table.calf,
        "foot_l": table.foot * 0.8,
        "ball_l": table.foot * 0.2,
        "thigh_r": table.hip_flap + table.thigh_minus_flap,
        "calf_r": table.calf,
        "foot_r": table.foot * 0.8,
        "ball_r": table.foot * 0.2,
    }
    values = [float(masses.get(name, 0.0)) for name in bone_names]
    total = sum(values)
    if total <= 0.0:
        raise ValueError("Body mass table produced zero total mass for this skeleton.")
    return [value / total for value in values]
