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
    return GAM("y ~ s(x1, k=10) + s(x2, k=10)").fit({"y": y, "x1": x1, "x2": x2})


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


# ---------------------------------------------------------------------------
# 2-D partial effects (te / ti)
# ---------------------------------------------------------------------------


RNG2 = np.random.default_rng(42)


def _fitted_te() -> GAM:
    n = 200
    x1 = RNG2.uniform(size=n)
    x2 = RNG2.uniform(size=n)
    y = np.sin(3 * x1) * np.cos(3 * x2) + 0.2 * RNG2.standard_normal(n)
    return GAM("y ~ te(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})


def _fitted_anova() -> GAM:
    n = 300
    x1 = np.random.default_rng(99).uniform(size=n)
    x2 = np.random.default_rng(99).uniform(high=1.0, size=n)
    y = np.sin(3 * x1) + np.cos(3 * x2) + 2 * x1 * x2 + 0.2 * np.random.default_rng(99).standard_normal(n)
    return GAM("y ~ s(x1, k=6) + s(x2, k=6) + ti(x1, x2, k=5)").fit(
        {"y": y, "x1": x1, "x2": x2}
    )


class TestPartialEffects2D:
    def test_te_returns_hconcat(self) -> None:
        model = _fitted_te()
        chart = model.plot()
        assert isinstance(chart, alt.HConcatChart)

    def test_te_spec_has_two_panels(self) -> None:
        model = _fitted_te()
        spec = model.plot().to_dict()
        assert len(spec.get("hconcat", [])) == 2

    def test_te_spec_valid(self) -> None:
        model = _fitted_te()
        spec = model.plot().to_dict()
        assert spec is not None

    def test_anova_mixed_1d_2d(self) -> None:
        model = _fitted_anova()
        chart = model.plot()
        assert isinstance(chart, alt.VConcatChart)
        spec = chart.to_dict()
        panels = spec.get("vconcat", [])
        assert len(panels) == 3

    def test_anova_1d_panels_are_layer(self) -> None:
        model = _fitted_anova()
        spec = model.plot().to_dict()
        panels = spec["vconcat"]
        assert "layer" in panels[0]
        assert "layer" in panels[1]

    def test_anova_2d_panel_is_hconcat(self) -> None:
        model = _fitted_anova()
        spec = model.plot().to_dict()
        panels = spec["vconcat"]
        assert "hconcat" in panels[2]

    def test_te_custom_n_points(self) -> None:
        model = _fitted_te()
        chart = model.plot(n_points=100)
        spec = chart.to_dict()
        assert spec is not None

    def test_te_custom_level(self) -> None:
        model = _fitted_te()
        chart = model.plot(level=0.99)
        assert chart.to_dict() is not None

    def test_te_heatmap_has_color(self) -> None:
        model = _fitted_te()
        spec = model.plot().to_dict()
        effect_spec = spec["hconcat"][0]
        encoding = effect_spec.get("encoding", {})
        assert "color" in encoding

    def test_te_poisson(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        mu = np.exp(0.5 + x1 * x2)
        y = rng.poisson(mu).astype(float)

        model = GAM("y ~ te(x1, x2, k=4)", family=Poisson()).fit(
            {"y": y, "x1": x1, "x2": x2}
        )
        chart = model.plot()
        assert chart.to_dict() is not None

    def test_ti_only(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.uniform(size=n)
        x2 = rng.uniform(size=n)
        y = x1 * x2 + 0.1 * rng.standard_normal(n)

        model = GAM("y ~ ti(x1, x2, k=5)").fit({"y": y, "x1": x1, "x2": x2})
        chart = model.plot()
        assert isinstance(chart, alt.HConcatChart)

    def test_se_panel_has_reds_scheme(self) -> None:
        model = _fitted_te()
        spec = model.plot().to_dict()
        se_spec = spec["hconcat"][1]
        color_enc = se_spec.get("encoding", {}).get("color", {})
        scale = color_enc.get("scale", {})
        assert scale.get("scheme") == "reds"

    def test_effect_panel_diverging_scale(self) -> None:
        model = _fitted_te()
        spec = model.plot().to_dict()
        effect_spec = spec["hconcat"][0]
        color_enc = effect_spec.get("encoding", {}).get("color", {})
        scale = color_enc.get("scale", {})
        assert scale.get("domainMid") == 0
