"""Smooth, fixed-port spiral delay cells."""

from collections.abc import Sequence
from itertools import pairwise
from math import cos, pi, sin
from typing import Protocol, Unpack

from .. import kdb
from ..cross_section import (
    AnyCrossSectionInput,
    CrossSectionSpecDict,
    DCrossSectionSpecDict,
)
from ..enclosure import LayerEnclosure, extrude_path_cross_section
from ..kcell import KCell
from ..layout import CellKWargs, KCLayout
from ..settings import Info
from ..typings import um
from .utils import boundary_from_shapes, cross_section_from_width

__all__ = [
    "SpiralDelayFactory",
    "minimum_spiral_length",
    "spiral_delay_factory",
]


class SpiralDelayFactory(Protocol):
    """Factory protocol for a two-port spiral delay cell."""

    def __call__(
        self,
        *,
        target_length: um,
        cross_section: str
        | AnyCrossSectionInput
        | CrossSectionSpecDict
        | DCrossSectionSpecDict
        | None = None,
        width: um | None = None,
        layer: kdb.LayerInfo | None = None,
        enclosure: LayerEnclosure | None = None,
        inner_radius: um = 10.0,
        turn_pitch: um = 23.0,
        arm_offset: um = 20.0,
        port_clearance: um = 10.0,
        max_turns: int = 20,
        points_per_turn: int = 128,
    ) -> KCell:
        """Create a spiral with a requested centerline length in micrometres."""
        ...


def _path_length(points: Sequence[kdb.DPoint]) -> float:
    """Return the sampled centerline length in micrometres."""
    return sum(float((p2 - p1).length()) for p1, p2 in pairwise(points))


def _spiral_core(
    *,
    turns: int,
    inner_radius: um,
    turn_pitch: um,
    arm_offset: um,
    points_per_turn: int,
) -> list[kdb.DPoint]:
    """Return the tangent-continuous double-spiral centerline.

    The two access straights are omitted. All dimensions are in micrometres.
    """
    if turns < 1:
        raise ValueError("turns must be at least 1")
    if inner_radius <= 0:
        raise ValueError("inner_radius must be positive")
    if turn_pitch <= 0:
        raise ValueError("turn_pitch must be positive")
    if arm_offset < 2 * inner_radius:
        raise ValueError(
            "arm_offset / 2 is the central turn radius and must be at least "
            "inner_radius"
        )
    if points_per_turn < 32:
        raise ValueError("points_per_turn must be at least 32")

    theta_max = 2 * pi * turns
    outer_radius = inner_radius + turns * turn_pitch
    radial_slope = turn_pitch / (2 * pi)
    samples = turns * points_per_turn

    inward_arm: list[kdb.DPoint] = []
    for i in range(samples + 1):
        theta = theta_max * i / samples
        radius = outer_radius - radial_slope * theta + radial_slope * sin(theta)
        inward_arm.append(
            kdb.DPoint(
                radius * cos(theta) - outer_radius,
                radius * sin(theta),
            )
        )

    inward_arm[0] = kdb.DPoint(0, 0)
    inward_arm[-1] = kdb.DPoint(inner_radius - outer_radius, 0)

    turn_radius = arm_offset / 2
    center_x = inner_radius - outer_radius + turn_radius
    half_samples = max(16, points_per_turn // 2)
    center_turn = [
        kdb.DPoint(
            center_x + turn_radius * cos(pi - pi * i / half_samples),
            turn_radius * sin(pi - pi * i / half_samples),
        )
        for i in range(1, half_samples + 1)
    ]

    outward_arm: list[kdb.DPoint] = []
    for i in range(1, samples + 1):
        theta = theta_max * (1 - i / samples)
        radius = (
            outer_radius - radial_slope * theta + radial_slope * sin(theta) + arm_offset
        )
        outward_arm.append(
            kdb.DPoint(
                radius * cos(theta) - outer_radius,
                radius * sin(theta),
            )
        )

    points = inward_arm + center_turn + outward_arm
    points[-1] = kdb.DPoint(arm_offset, 0)
    return points


def minimum_spiral_length(
    *,
    turns: int = 1,
    inner_radius: um = 10.0,
    turn_pitch: um = 23.0,
    arm_offset: um = 20.0,
    port_clearance: um = 10.0,
    points_per_turn: int = 128,
) -> float:
    """Return the minimum centerline length in micrometres."""
    if port_clearance < 0:
        raise ValueError("port_clearance must be non-negative")
    core = _spiral_core(
        turns=turns,
        inner_radius=inner_radius,
        turn_pitch=turn_pitch,
        arm_offset=arm_offset,
        points_per_turn=points_per_turn,
    )
    outer_radius = inner_radius + turns * turn_pitch
    access_length = outer_radius + arm_offset + port_clearance
    return _path_length(core) + 2 * access_length


def spiral_delay_factory(
    kcl: KCLayout,
    *,
    port_type: str = "optical",
    **cell_kwargs: Unpack[CellKWargs],
) -> SpiralDelayFactory:
    """Return a cached fixed-port spiral-delay PCell factory.

    The generated cell has two ports at ``(0, 0)`` and ``(arm_offset, 0)``.
    Both ports face south. Changing ``target_length`` only changes geometry
    above the ports, which lets a router size the delay after routing its
    connector segments.
    """
    cell_kwargs.setdefault("basename", "spiral_delay")
    basename = cell_kwargs["basename"]
    cell = kcl.cell(output_type=KCell, **cell_kwargs)

    @cell
    def _spiral_delay(
        cross_section: str | AnyCrossSectionInput,
        target_length: um,
        inner_radius: um = 10.0,
        turn_pitch: um = 23.0,
        arm_offset: um = 20.0,
        port_clearance: um = 10.0,
        max_turns: int = 20,
        points_per_turn: int = 128,
    ) -> KCell:
        if target_length <= 0:
            raise ValueError("target_length must be positive")
        if port_clearance < 0:
            raise ValueError("port_clearance must be non-negative")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        c = kcl.kcell()
        xs = kcl.get_base_cross_section(cross_section)
        width = kcl.to_um(xs.width)
        if turn_pitch <= arm_offset + width:
            raise ValueError(
                "turn_pitch must exceed arm_offset + waveguide width to leave "
                "centerline clearance"
            )

        chosen: tuple[list[kdb.DPoint], float, float, int] | None = None
        for turns in range(1, max_turns + 1):
            core = _spiral_core(
                turns=turns,
                inner_radius=inner_radius,
                turn_pitch=turn_pitch,
                arm_offset=arm_offset,
                points_per_turn=points_per_turn,
            )
            core_length = _path_length(core)
            outer_radius = inner_radius + turns * turn_pitch
            minimum_access = outer_radius + arm_offset + port_clearance
            if core_length + 2 * minimum_access <= target_length:
                chosen = core, core_length, minimum_access, turns
            else:
                break

        if chosen is None:
            minimum = minimum_spiral_length(
                inner_radius=inner_radius,
                turn_pitch=turn_pitch,
                arm_offset=arm_offset,
                port_clearance=port_clearance,
                points_per_turn=points_per_turn,
            )
            raise ValueError(
                f"target_length={target_length:.3f} um is too short; "
                f"minimum is {minimum:.3f} um"
            )

        core, core_length, minimum_access, turns = chosen
        access_length = (
            minimum_access + (target_length - core_length - 2 * minimum_access) / 2
        )
        points = [kdb.DPoint(0, 0)]
        points.extend(kdb.DPoint(point.x, point.y + access_length) for point in core)
        points.append(kdb.DPoint(arm_offset, 0))

        extrude_path_cross_section(
            c,
            points,
            xs,
            start_angle=90,
            end_angle=-90,
        )
        c.create_port(
            name="o1",
            trans=kdb.Trans.R270,
            cross_section=xs,
            port_type=port_type,
        )
        c.create_port(
            name="o2",
            trans=kdb.Trans(
                3,
                False,
                kcl.to_dbu(arm_offset),
                0,
            ),
            cross_section=xs,
            port_type=port_type,
        )
        c.info = Info(
            length_dbu=kcl.to_dbu(target_length),
            length_um=_path_length(points),
            turns=turns,
        )
        boundary = boundary_from_shapes(c)
        if boundary is not None:
            c.boundary = boundary
        return c

    @kcl.generic_factory(name=basename)
    def spiral_delay(
        *,
        target_length: um,
        cross_section: str
        | AnyCrossSectionInput
        | CrossSectionSpecDict
        | DCrossSectionSpecDict
        | None = None,
        width: um | None = None,
        layer: kdb.LayerInfo | None = None,
        enclosure: LayerEnclosure | None = None,
        inner_radius: um = 10.0,
        turn_pitch: um = 23.0,
        arm_offset: um = 20.0,
        port_clearance: um = 10.0,
        max_turns: int = 20,
        points_per_turn: int = 128,
    ) -> KCell:
        if cross_section is None:
            if width is None or layer is None:
                raise ValueError(
                    "Provide a cross_section, or width and layer (legacy call)."
                )
            xs = cross_section_from_width(kcl, kcl.to_dbu(width), layer, enclosure)
        else:
            xs = kcl.get_icross_section(cross_section)
        return _spiral_delay(
            cross_section=xs,
            target_length=target_length,
            inner_radius=inner_radius,
            turn_pitch=turn_pitch,
            arm_offset=arm_offset,
            port_clearance=port_clearance,
            max_turns=max_turns,
            points_per_turn=points_per_turn,
        )

    return spiral_delay
