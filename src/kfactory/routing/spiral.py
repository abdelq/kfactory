"""Path-length matching with smooth, fixed-site spiral delay cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, computed_field

from .. import kdb
from ..instance import Instance  # noqa: TC001
from ..kcell import KCell  # noqa: TC001
from ..port import Port, ProtoPort
from ..typings import dbu  # noqa: TC001
from .generic import ManhattanRoute  # noqa: TC001
from .optical import route_bundle

if TYPE_CHECKING:
    from ..factories import StraightFactoryDBU
    from ..factories.spiral import SpiralDelayFactory

__all__ = ["SpiralDelayRoute", "SpiralSite", "route_bundle_spiral_delay"]


class SpiralSite(BaseModel, frozen=True):
    """Fixed site for one spiral delay cell, in database units."""

    x: dbu
    y: dbu
    rotation: Literal[0, 1, 2, 3] = 0
    mirror: bool = False

    @property
    def trans(self) -> kdb.Trans:
        """Return the site's integer transformation."""
        return kdb.Trans(self.rotation, self.mirror, self.x, self.y)


class SpiralDelayRoute(BaseModel, arbitrary_types_allowed=True):
    """Composite route made from two connectors and one spiral delay."""

    input_route: ManhattanRoute
    delay_instance: Instance
    delay_length: dbu
    output_route: ManhattanRoute

    @computed_field
    @property
    def length(self) -> float:
        """Total connector-plus-delay length in database units."""
        return (
            float(self.input_route.length)
            + self.delay_length
            + float(self.output_route.length)
        )

    @property
    def start_port(self) -> Port:
        return self.input_route.start_port

    @property
    def end_port(self) -> Port:
        return self.output_route.end_port

    @property
    def instances(self) -> list[Instance]:
        return [
            *self.input_route.instances,
            self.delay_instance,
            *self.output_route.instances,
        ]


def route_bundle_spiral_delay(
    c: KCell,
    ports: Sequence[Sequence[ProtoPort[Any]]],
    *,
    separation: dbu,
    straight_factory: StraightFactoryDBU,
    bend90_cell: KCell,
    spiral_factory: SpiralDelayFactory,
    sites: Sequence[SpiralSite | Mapping[str, Any]],
    target_length: dbu | None = None,
    inner_radius: dbu = 10_000,
    turn_pitch: dbu = 23_000,
    arm_offset: dbu = 20_000,
    port_clearance: dbu = 10_000,
    max_turns: int = 20,
    points_per_turn: int = 128,
    taper_cell: KCell | None = None,
    min_straight_taper: dbu = 0,
    place_port_type: str = "optical",
    place_allow_small_routes: bool = False,
    collision_check_layers: Sequence[kdb.LayerInfo] | None = None,
    on_collision: Literal["error", "show_error"] | None = "show_error",
    on_placer_error: Literal["error", "show_error"] | None = "show_error",
    allow_width_mismatch: bool | None = None,
    allow_layer_mismatch: bool | None = None,
    allow_type_mismatch: bool | None = None,
    route_width: dbu | list[dbu] | None = None,
    purpose: str | None = "routing",
) -> list[SpiralDelayRoute]:
    """Route and length-match a bundle using smooth spiral delay cells.

    This function has the ``(cell, nested_ports, **settings)`` shape expected
    by [`KCLayout.routing_strategy`][kfactory.layout.KCLayout.routing_strategy].
    A PDK wrapper can close over the three cell factories and expose the
    remaining JSON-serializable settings to a schematic.

    Each item in ``ports`` must contain the source and destination port for one
    channel, with one corresponding item in ``sites``. Connector routes are
    materialized first; the residual length at each site is then implemented by
    a spiral whose external ports do not move as its length changes.
    """
    from ..factories.spiral import minimum_spiral_length

    parsed_sites = [
        site if isinstance(site, SpiralSite) else SpiralSite.model_validate(site)
        for site in sites
    ]
    if len(ports) != len(parsed_sites):
        raise ValueError(
            "Spiral delay routing requires one site per routed channel: "
            f"got {len(parsed_sites)} sites for {len(ports)} channels"
        )
    if not ports:
        return []

    net_ports: list[tuple[Port, Port]] = []
    for channel, channel_ports in enumerate(ports):
        if len(channel_ports) != 2:
            raise ValueError(
                f"Spiral delay channel {channel} must contain exactly two ports"
            )
        start_port, end_port = channel_ports
        if not isinstance(start_port, Port) or not isinstance(end_port, Port):
            raise TypeError(
                "route_bundle_spiral_delay currently supports integer-unit KCell "
                "ports only"
            )
        net_ports.append((start_port, end_port))

    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    minimum_delay_um = minimum_spiral_length(
        inner_radius=c.kcl.to_um(inner_radius),
        turn_pitch=c.kcl.to_um(turn_pitch),
        arm_offset=c.kcl.to_um(arm_offset),
        port_clearance=c.kcl.to_um(port_clearance),
        points_per_turn=points_per_turn,
    )
    minimum_delay = ceil(minimum_delay_um / c.kcl.dbu)

    site_inputs: list[Port] = []
    site_outputs: list[Port] = []
    for channel, ((start_port, _), site) in enumerate(
        zip(net_ports, parsed_sites, strict=True)
    ):
        site_inputs.append(
            Port(
                name=f"spiral_site_{channel}_o1",
                trans=site.trans * kdb.Trans.R270,
                cross_section=start_port.cross_section,
                port_type=place_port_type,
                kcl=c.kcl,
            )
        )
        site_outputs.append(
            Port(
                name=f"spiral_site_{channel}_o2",
                trans=site.trans
                * kdb.Trans(
                    3,
                    False,
                    arm_offset,
                    0,
                ),
                cross_section=start_port.cross_section,
                port_type=place_port_type,
                kcl=c.kcl,
            )
        )

    common_route_kwargs: dict[str, Any] = {
        "separation": separation,
        "straight_factory": straight_factory,
        "bend90_cell": bend90_cell,
        "taper_cell": taper_cell,
        "min_straight_taper": min_straight_taper,
        "place_port_type": place_port_type,
        "place_allow_small_routes": place_allow_small_routes,
        "collision_check_layers": collision_check_layers,
        "on_collision": on_collision,
        "on_placer_error": on_placer_error,
        "allow_width_mismatch": allow_width_mismatch,
        "allow_layer_mismatch": allow_layer_mismatch,
        "allow_type_mismatch": allow_type_mismatch,
        "route_width": route_width,
        "purpose": purpose,
    }
    input_routes = route_bundle(
        c,
        [start for start, _ in net_ports],
        site_inputs,
        **common_route_kwargs,
    )
    output_routes = route_bundle(
        c,
        site_outputs,
        [end for _, end in net_ports],
        **common_route_kwargs,
    )
    if len(input_routes) != len(net_ports) or len(output_routes) != len(net_ports):
        raise RuntimeError("Connector routing did not return one result per channel")

    connector_lengths = [
        float(input_route.length) + float(output_route.length)
        for input_route, output_route in zip(input_routes, output_routes, strict=True)
    ]
    minimum_target = ceil(
        max(connector_length + minimum_delay for connector_length in connector_lengths)
    )
    if target_length is None:
        target_length = minimum_target
    elif target_length < minimum_target:
        raise ValueError(
            f"target_length={target_length} dbu is too short; "
            f"the routed connectors and minimum spirals require {minimum_target} dbu"
        )

    results: list[SpiralDelayRoute] = []
    for (
        (start_port, _),
        site,
        input_route,
        output_route,
        connector_length,
    ) in zip(
        net_ports,
        parsed_sites,
        input_routes,
        output_routes,
        connector_lengths,
        strict=True,
    ):
        delay_length = round(target_length - connector_length)
        delay_cell = spiral_factory(
            target_length=c.kcl.to_um(delay_length),
            cross_section=start_port.cross_section,
            inner_radius=c.kcl.to_um(inner_radius),
            turn_pitch=c.kcl.to_um(turn_pitch),
            arm_offset=c.kcl.to_um(arm_offset),
            port_clearance=c.kcl.to_um(port_clearance),
            max_turns=max_turns,
            points_per_turn=points_per_turn,
        )
        delay_instance = c << delay_cell
        delay_instance.transform(site.trans)
        results.append(
            SpiralDelayRoute(
                input_route=input_route,
                delay_instance=delay_instance,
                delay_length=delay_length,
                output_route=output_route,
            )
        )

    return results
