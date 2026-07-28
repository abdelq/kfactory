"""Length-match photonic routes with smooth, fixed-site spiral delays.

Unlike ``PathLengthMatch``, which edits a Manhattan backbone to create
meanders, the dedicated spiral strategy routes to explicit delay sites,
measures those connectors, and sizes one smooth spiral per channel to reach a
common total length.
"""
# ruff: noqa: INP001

from __future__ import annotations

import kfactory as kf


def build_matched_delay_bank(
    kcl: kf.KCLayout = kf.kcl,
    layer: kf.kdb.LayerInfo | None = None,
    width: float = 0.5,
    inner_radius: float = 10.0,
    turn_pitch: float = 23.0,
    arm_offset: float = 20.0,
    port_clearance: float = 10.0,
) -> tuple[kf.KCell, list[kf.SpiralDelayRoute]]:
    """Build three connector-plus-spiral routes with equal total lengths."""
    if layer is None:
        layer = kf.kdb.LayerInfo(1, 0)
    bend90 = kf.factories.euler.bend_euler_factory(kcl=kcl)(
        width=width,
        radius=inner_radius,
        layer=layer,
        angle=90,
    )
    straight = kf.factories.straight.straight_dbu_factory(kcl=kcl)

    def straight_factory(width: int, length: int) -> kf.KCell:
        return straight(width=width, length=length, layer=layer)

    spiral_factory = kf.factories.spiral.spiral_delay_factory(kcl)

    top = kcl.kcell("matched_spiral_delays")
    anchor_y = 250.0
    source_y = [0.0, -50.0, -120.0]
    output_y = [-50.0, -20.0, -80.0]
    sites = [
        kf.SpiralSite(
            x=kcl.to_dbu(channel * 500.0),
            y=kcl.to_dbu(anchor_y),
        )
        for channel in range(3)
    ]
    nets: list[tuple[kf.Port, kf.Port]] = []
    for channel, (site, source_y_um, output_y_um) in enumerate(
        zip(sites, source_y, output_y, strict=True)
    ):
        source = kf.Port(
            name=f"in_{channel}",
            trans=kf.kdb.Trans(
                1,
                False,
                site.x,
                kcl.to_dbu(source_y_um),
            ),
            width=kcl.to_dbu(width),
            layer_info=layer,
            kcl=kcl,
        )
        output = kf.Port(
            name=f"out_{channel}",
            trans=kf.kdb.Trans(
                1,
                False,
                site.x + kcl.to_dbu(arm_offset),
                kcl.to_dbu(output_y_um),
            ),
            width=kcl.to_dbu(width),
            layer_info=layer,
            kcl=kcl,
        )
        top.add_port(port=source)
        top.add_port(port=output)
        nets.append((source, output))

    routes = kf.routing.spiral.route_bundle_spiral_delay(
        top,
        nets,
        separation=kcl.to_dbu(5.0),
        straight_factory=straight_factory,
        bend90_cell=bend90,
        spiral_factory=spiral_factory,
        sites=sites,
        inner_radius=kcl.to_dbu(inner_radius),
        turn_pitch=kcl.to_dbu(turn_pitch),
        arm_offset=kcl.to_dbu(arm_offset),
        port_clearance=kcl.to_dbu(port_clearance),
        on_collision=None,
    )
    return top, routes


if __name__ == "__main__":
    cell, routes = build_matched_delay_bank()
    cell.show()
