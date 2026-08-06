"""Tests for whittaker.plotting (Altair-based GAM diagnostics)."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.families.binomial import Binomial
from whittaker.families.poisson import Poisson
from whittaker.gam import GAM

alt = pytest.importorskip("altair")



RNG = np.random.default_rng(23)


def _fitted_gaussian() -> GAM:
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + RNG.normal(0, 0.2, n)
    return GAM("y ~ s(x, k=10)").fit({"y": y, "x": x})


def _fitted_multi() -> GAM:
    n = 300
    x1 = np.linspace(0, 2 * np.pi, n)
    x2 = np.linspace(0, 1, n)
    y = np.sin(x1) + 0.5 * x2 + RNG.normal(0, 0.2, n)
    return GAM("y ~ s(x1, k=10) + s(x2, k=10)").fit(
        {"y": y, "x1": x1, "x2": x2}
    )


def _fitted_binomial() -> GAM:
    rng = np.random.default_rng(42)
    n = 300
    x = np.linspace(-3, 3, n)
    p_true = 1.0 / (1.0 + np.exp(-np.sin(x)))
    y = rng.binomial(1, p_true, n).astype(float)
    return GAM("y ~ s(x, k=10)", family=Binomial()).fit({"y": y, "x": x})


def _fitted_poisson() -> GAM:
    rng = np.random.default_rng(42)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.5 * np.sin(x))
    y = rng.poisson(mu).astype(float)
    return GAM("y ~ s(x, k=10)", family=Poisson()).fit({"y": y, "x": x})


# ---------------------------------------------------------------------------
# partial_effects (model.plot())
# ---------------------------------------------------------------------------


class TestPartialEffects:
    def test_returns_altair_chart(self) -> None:

        model = _fitted_gaussian()
        chart = model.plot()
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.VConcatChart))

    def test_single_smooth_not_vconcatted(self) -> None:

        model = _fitted_gaussian()
        chart = model.plot()
        assert not isinstance(chart, alt.VConcatChart)

    def test_multi_smooth_vconcatted(self) -> None:

        model = _fitted_multi()
        chart = model.plot()
        assert isinstance(chart, alt.VConcatChart)

    def test_custom_n_points(self) -> None:
        model = _fitted_gaussian()
        chart = model.plot(n_points=50)
        spec = chart.to_dict()
        data = spec.get("data", spec.get("datasets", {}))
        assert data is not None

    def test_custom_level(self) -> None:
        model = _fitted_gaussian()
        chart_wide = model.plot(level=0.99)
        chart_narrow = model.plot(level=0.80)
        assert chart_wide.to_dict() != chart_narrow.to_dict()

    def test_unfitted_raises(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.plot()

    def test_binomial_partial_effects(self) -> None:
        model = _fitted_binomial()
        chart = model.plot()
        assert chart.to_dict() is not None

    def test_poisson_partial_effects(self) -> None:
        model = _fitted_poisson()
        chart = model.plot()
        assert chart.to_dict() is not None

    @pytest.mark.parametrize("bs", ["tp", "cr", "ps"])
    def test_basis_types(self, bs: str) -> None:
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y = np.sin(x) + RNG.normal(0, 0.2, n)
        model = GAM(f"y ~ s(x, bs='{bs}', k=10)").fit({"y": y, "x": x})
        chart = model.plot()
        assert chart.to_dict() is not None


# ---------------------------------------------------------------------------
# check (model.check())
# ---------------------------------------------------------------------------


class TestCheck:
    def test_returns_altair_vconcatchart(self) -> None:

        model = _fitted_gaussian()
        chart = model.check()
        assert isinstance(chart, alt.VConcatChart)

    def test_has_four_panels(self) -> None:
        model = _fitted_gaussian()
        spec = model.check().to_dict()
        vconcat = spec.get("vconcat", [])
        assert len(vconcat) == 2
        for row in vconcat:
            assert len(row.get("hconcat", [])) == 2

    def test_unfitted_raises(self) -> None:
        model = GAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.check()

    def test_binomial_check(self) -> None:
        model = _fitted_binomial()
        chart = model.check()
        assert chart.to_dict() is not None

    def test_poisson_check(self) -> None:
        model = _fitted_poisson()
        chart = model.check()
        assert chart.to_dict() is not None

    def test_multi_smooth_check(self) -> None:
        model = _fitted_multi()
        chart = model.check()
        assert chart.to_dict() is not None
