from collections.abc import Callable

import pytest

import kfactory as kf
from tests.conftest import Layers


def test_spiral_delay_factory_has_fixed_ports(
    kcl: kf.KCLayout,
    layers: Layers,
) -> None:
    factory = kf.factories.spiral.spiral_delay_factory(kcl)
    target_length = 600.0
    arm_offset = 20.0

    delay = factory(
        target_length=target_length,
        width=0.5,
        layer=layers.WG,
        arm_offset=arm_offset,
    )

    assert delay.ports["o1"].trans == kf.kdb.Trans.R270
    assert delay.ports["o2"].trans == kf.kdb.Trans(3, False, kcl.to_dbu(arm_offset), 0)
    assert delay.info["length_dbu"] == kcl.to_dbu(target_length)
    assert float(delay.info["length_um"]) == pytest.approx(target_length)
    assert int(delay.info["turns"]) >= 1


def test_spiral_delay_rejects_too_short_target(
    kcl: kf.KCLayout,
    layers: Layers,
) -> None:
    factory = kf.factories.spiral.spiral_delay_factory(kcl)
    minimum = kf.factories.spiral.minimum_spiral_length()

    with pytest.raises(ValueError, match="too short"):
        factory(
            target_length=minimum - 1,
            width=0.5,
            layer=layers.WG,
        )


def test_spiral_delay_routes_to_common_length(
    kcl: kf.KCLayout,
    layers: Layers,
    straight_factory_dbu: Callable[..., kf.KCell],
    bend90: kf.KCell,
) -> None:
    c = kcl.kcell("spiral_delay_match")
    arm_offset = 20_000
    sites = [
        kf.SpiralSite(x=0, y=250_000),
        kf.SpiralSite(x=500_000, y=250_000),
        kf.SpiralSite(x=1_000_000, y=250_000),
    ]
    source_y = [0, -50_000, -120_000]
    output_y = [-50_000, -20_000, -80_000]
    nets: list[tuple[kf.Port, kf.Port]] = []
    for channel, (site, sy, ey) in enumerate(
        zip(sites, source_y, output_y, strict=True)
    ):
        nets.append(
            (
                kf.Port(
                    name=f"in_{channel}",
                    width=500,
                    layer_info=layers.WG,
                    trans=kf.kdb.Trans(1, False, site.x, sy),
                    kcl=kcl,
                ),
                kf.Port(
                    name=f"out_{channel}",
                    width=500,
                    layer_info=layers.WG,
                    trans=kf.kdb.Trans(
                        1,
                        False,
                        site.x + arm_offset,
                        ey,
                    ),
                    kcl=kcl,
                ),
            )
        )

    routes = kf.routing.spiral.route_bundle_spiral_delay(
        c,
        nets,
        separation=5_000,
        straight_factory=straight_factory_dbu,
        bend90_cell=bend90,
        spiral_factory=kf.factories.spiral.spiral_delay_factory(kcl),
        sites=[site.model_dump() for site in sites],
        arm_offset=arm_offset,
        on_collision=None,
        on_placer_error="error",
    )

    lengths = [route.length for route in routes]
    assert len(routes) == len(nets)
    assert max(lengths) - min(lengths) <= 1
    for route, site in zip(routes, sites, strict=True):
        assert route.delay_instance.ports["o1"].trans == site.trans * kf.kdb.Trans.R270
        assert route.delay_instance.cell.info["length_dbu"] == route.delay_length
        assert route.length == pytest.approx(lengths[0], abs=1)


def test_spiral_delay_requires_one_site_per_channel(
    kcl: kf.KCLayout,
    straight_factory_dbu: Callable[..., kf.KCell],
    bend90: kf.KCell,
) -> None:
    with pytest.raises(ValueError, match="one site per routed channel"):
        kf.routing.spiral.route_bundle_spiral_delay(
            kcl.kcell(),
            [],
            separation=5_000,
            straight_factory=straight_factory_dbu,
            bend90_cell=bend90,
            spiral_factory=kf.factories.spiral.spiral_delay_factory(kcl),
            sites=[{"x": 0, "y": 0}],
        )


def test_spiral_delay_accepts_empty_bundle(
    kcl: kf.KCLayout,
    straight_factory_dbu: Callable[..., kf.KCell],
    bend90: kf.KCell,
) -> None:
    routes = kf.routing.spiral.route_bundle_spiral_delay(
        kcl.kcell(),
        [],
        separation=5_000,
        straight_factory=straight_factory_dbu,
        bend90_cell=bend90,
        spiral_factory=kf.factories.spiral.spiral_delay_factory(kcl),
        sites=[],
    )

    assert routes == []
