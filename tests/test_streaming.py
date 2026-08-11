"""Tests for streaming / online GAMs."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.gam import GAM
from whittaker.streaming import StreamingGAM, StreamingSnapshot


@pytest.fixture
def sin_data():
    rng = np.random.default_rng(23)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    return {"x": x, "y": y}


@pytest.fixture
def sin_batches():
    rng = np.random.default_rng(23)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + rng.normal(0, 0.3, n)
    batch_size = 100
    batches = []
    for i in range(0, n, batch_size):
        batches.append({"x": x[i : i + batch_size], "y": y[i : i + batch_size]})
    return batches


class TestStreamingGAMInit:
    def test_default_init(self):
        model = StreamingGAM("y ~ s(x)")
        assert not model.is_initialised
        assert not model.is_solved
        assert model.n_obs == 0
        assert model.n_batches == 0

    def test_repr_unfitted(self):
        model = StreamingGAM("y ~ s(x)")
        assert "unfitted" in repr(model)

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="decay must be in"):
            StreamingGAM("y ~ s(x)", decay=0.0)
        with pytest.raises(ValueError, match="decay must be in"):
            StreamingGAM("y ~ s(x)", decay=1.5)


class TestPartialFit:
    def test_first_batch_initialises(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        assert model.is_initialised
        assert model.n_obs == 100
        assert model.n_batches == 1

    def test_auto_solves_on_first_batch(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        assert model.is_solved

    def test_accumulates_batches(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        assert model.n_obs == 300
        assert model.n_batches == 3

    def test_method_chaining(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        result = model.partial_fit(sin_batches[0])
        assert result is model


class TestSolve:
    def test_solve_updates_coefficients(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        coef_before = model.coefficients.copy()
        model.partial_fit(sin_batches[1])
        model.solve()
        coef_after = model.coefficients
        assert not np.allclose(coef_before, coef_after)

    def test_solve_returns_self(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        result = model.solve()
        assert result is model

    def test_solve_without_data_raises(self):
        model = StreamingGAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="No data"):
            model.solve()

    def test_solve_with_reestimate(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve(reestimate_smoothing=True)
        assert model.is_solved
        assert len(model.smoothing_params) > 0

    def test_edf_reasonable(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        assert 1.0 < model.edf_total < 20.0

    def test_scale_positive(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        assert model.scale > 0


class TestPredict:
    def test_predict_shape(self, sin_data, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        pred = model.predict(sin_data)
        assert pred.values.shape == (300,)

    def test_predict_with_se(self, sin_data, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        pred = model.predict(sin_data, se=True)
        assert pred.se is not None
        assert pred.se.shape == (300,)
        assert np.all(pred.se >= 0)

    def test_predict_before_solve_raises(self, sin_batches):
        model = StreamingGAM("y ~ s(x)", smoothing_params=[1.0])
        with pytest.raises(RuntimeError, match="not been solved"):
            model.predict(sin_batches[0])

    def test_predict_finite(self, sin_data, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        pred = model.predict(sin_data)
        assert np.all(np.isfinite(pred.values))

    def test_predict_captures_sin(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        x_test = np.linspace(0.5, 2 * np.pi - 0.5, 50)
        pred = model.predict({"x": x_test})
        expected = np.sin(x_test)
        corr = np.corrcoef(pred.values, expected)[0, 1]
        assert corr > 0.9


class TestAgreementWithGAM:
    def test_single_batch_agrees(self, sin_data):
        gam = GAM("y ~ s(x)")
        gam.fit(sin_data)

        sgam = StreamingGAM("y ~ s(x)")
        sgam.partial_fit(sin_data)
        sgam.solve()

        x_test = np.linspace(0.5, 2 * np.pi - 0.5, 50)
        gam_pred = gam.predict({"x": x_test}).values
        sgam_pred = sgam.predict({"x": x_test}).values
        np.testing.assert_allclose(sgam_pred, gam_pred, atol=0.2)


class TestDecay:
    def test_decay_downweights_old(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y1 = np.sin(x) + rng.normal(0, 0.2, n)
        y2 = 2 * np.sin(x) + rng.normal(0, 0.2, n)

        model_decay = StreamingGAM("y ~ s(x)", decay=0.5)
        model_decay.partial_fit({"x": x, "y": y1})
        model_decay.partial_fit({"x": x, "y": y2})
        model_decay.solve()

        model_nodecay = StreamingGAM("y ~ s(x)")
        model_nodecay.partial_fit({"x": x, "y": y1})
        model_nodecay.partial_fit({"x": x, "y": y2})
        model_nodecay.solve()

        x_test = np.linspace(0.5, 2 * np.pi - 0.5, 20)
        pred_decay = model_decay.predict({"x": x_test}).values
        pred_nodecay = model_nodecay.predict({"x": x_test}).values

        assert not np.allclose(pred_decay, pred_nodecay, atol=0.05)

    def test_decay_recent_data_dominates(self):
        rng = np.random.default_rng(23)
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        y_old = np.zeros(n) + rng.normal(0, 0.1, n)
        y_new = np.sin(x) + rng.normal(0, 0.1, n)

        model = StreamingGAM("y ~ s(x)", decay=0.1)
        model.partial_fit({"x": x, "y": y_old})
        model.partial_fit({"x": x, "y": y_new})
        model.solve()

        x_test = np.array([np.pi / 2])
        pred = model.predict({"x": x_test}).values[0]
        assert pred > 0.3


class TestShouldRefit:
    def test_not_initialised(self):
        model = StreamingGAM("y ~ s(x)")
        assert not model.should_refit()

    def test_after_few_batches(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        assert not model.should_refit(min_batches=5)

    def test_after_enough_batches(self):
        rng = np.random.default_rng(23)
        model = StreamingGAM("y ~ s(x)")
        for i in range(12):
            n = 30
            x = rng.uniform(0, 2 * np.pi, n)
            y = np.sin(x) + rng.normal(0, 0.3, n)
            model.partial_fit({"x": x, "y": y})
        assert model.should_refit(min_batches=10)


class TestSmoothingHistory:
    def test_history_grows(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        model.solve()
        assert len(model.smoothing_history()) == 1
        model.partial_fit(sin_batches[1])
        model.solve()
        assert len(model.smoothing_history()) == 2

    def test_snapshots_are_copies(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        model.solve()
        h = model.smoothing_history()
        assert isinstance(h[0], StreamingSnapshot)
        assert h[0].n_obs == 100

    def test_history_tracks_progress(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
            model.solve()
        h = model.smoothing_history()
        assert h[0].n_obs < h[-1].n_obs
        assert h[0].n_batches < h[-1].n_batches


class TestReset:
    def test_reset_clears_state(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        model.solve()
        model.reset()
        assert model.n_obs == 0
        assert model.n_batches == 0
        assert not model.is_solved
        assert model.is_initialised

    def test_reset_before_init_raises(self):
        model = StreamingGAM("y ~ s(x)")
        with pytest.raises(RuntimeError, match="before initialisation"):
            model.reset()

    def test_can_refit_after_reset(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        model.partial_fit(sin_batches[0])
        model.solve()
        model.reset()
        model.partial_fit(sin_batches[1])
        model.solve()
        assert model.is_solved
        assert model.n_obs == 100


class TestSummary:
    def test_summary_content(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        s = model.summary()
        assert "StreamingGAM" in s
        assert "N obs" in s
        assert "N batches" in s
        assert "EDF" in s

    def test_repr_solved(self, sin_batches):
        model = StreamingGAM("y ~ s(x)")
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve()
        r = repr(model)
        assert "fitted" in r
        assert "n=300" in r


class TestFixedSmoothingParams:
    def test_fixed_sp(self, sin_batches):
        model = StreamingGAM("y ~ s(x)", smoothing_params=[1.0])
        model.partial_fit(sin_batches[0])
        model.solve()
        assert model.smoothing_params == [1.0]

    def test_fixed_sp_no_reestimate(self, sin_batches):
        model = StreamingGAM("y ~ s(x)", smoothing_params=[1.0])
        for batch in sin_batches:
            model.partial_fit(batch)
        model.solve(reestimate_smoothing=True)
        assert model.smoothing_params == [1.0]


class TestTwoSmooths:
    def test_two_smooth_streaming(self):
        rng = np.random.default_rng(23)
        n = 300
        x1 = np.linspace(0, 2 * np.pi, n)
        x2 = rng.uniform(0, 1, n)
        y = np.sin(x1) + 2 * x2 + rng.normal(0, 0.3, n)

        model = StreamingGAM("y ~ s(x1) + s(x2)")
        batch_size = 100
        for i in range(0, n, batch_size):
            batch = {
                "x1": x1[i : i + batch_size],
                "x2": x2[i : i + batch_size],
                "y": y[i : i + batch_size],
            }
            model.partial_fit(batch)
        model.solve()

        x1_test = np.linspace(0.5, 2 * np.pi - 0.5, 30)
        x2_test = np.full(30, 0.5)
        pred = model.predict({"x1": x1_test, "x2": x2_test})
        assert pred.values.shape == (30,)
        assert np.all(np.isfinite(pred.values))
