"""Tests for the built-in datasets module."""

from __future__ import annotations

import numpy as np
import pytest

from whittaker.datasets import list_datasets, load_dataset


class TestListDatasets:
    def test_returns_non_empty_list(self):
        rows = list_datasets()
        assert isinstance(rows, list)
        assert len(rows) > 0

    def test_each_entry_has_name_key(self):
        rows = list_datasets()
        for row in rows:
            assert "name" in row

    def test_expected_dataset_names_present(self):
        names = {row["name"] for row in list_datasets()}
        expected = {
            "mcycle",
            "co2",
            "fish",
            "credit",
            "wages",
            "proportions",
            "meuse",
            "survival",
            "abalone",
            "climate",
        }
        assert expected <= names

    def test_each_entry_has_required_metadata_keys(self):
        required = {"name", "description", "variables", "family", "note"}
        for row in list_datasets():
            assert required <= set(row.keys()), f"Missing keys in entry for {row.get('name')}"


class TestLoadDataset:
    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_unknown_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("unknown")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_dataset(self, name: str, expected_keys: set[str]) -> None:
        data = load_dataset(name)
        assert isinstance(data, dict), f"{name}: result is not a dict"
        assert set(data.keys()) == expected_keys, (
            f"{name}: expected keys {expected_keys}, got {set(data.keys())}"
        )
        for key, arr in data.items():
            assert isinstance(arr, np.ndarray), f"{name}[{key!r}] is not an ndarray"
            assert np.all(np.isfinite(arr)), f"{name}[{key!r}] contains non-finite values"

    # ------------------------------------------------------------------
    # Individual dataset tests
    # ------------------------------------------------------------------

    def test_mcycle_keys_and_arrays(self):
        self._check_dataset("mcycle", {"times", "accel"})

    def test_mcycle_size(self):
        data = load_dataset("mcycle")
        assert data["times"].shape == (133,)
        assert data["accel"].shape == (133,)

    def test_co2_keys_and_arrays(self):
        self._check_dataset("co2", {"t", "year", "month", "co2"})

    def test_co2_size(self):
        data = load_dataset("co2")
        assert data["co2"].shape == (504,)

    def test_fish_keys_and_arrays(self):
        self._check_dataset("fish", {"temperature", "depth", "count"})

    def test_fish_size(self):
        data = load_dataset("fish")
        assert data["count"].shape == (300,)

    def test_fish_count_non_negative(self):
        data = load_dataset("fish")
        assert np.all(data["count"] >= 0)

    def test_credit_keys_and_arrays(self):
        self._check_dataset("credit", {"income", "debt_ratio", "age", "default"})

    def test_credit_size(self):
        data = load_dataset("credit")
        assert data["default"].shape == (1000,)

    def test_credit_default_binary(self):
        data = load_dataset("credit")
        assert set(np.unique(data["default"])) <= {0.0, 1.0}

    def test_wages_keys_and_arrays(self):
        self._check_dataset("wages", {"age", "experience", "wage"})

    def test_wages_size(self):
        data = load_dataset("wages")
        assert data["wage"].shape == (800,)

    def test_wages_positive(self):
        data = load_dataset("wages")
        assert np.all(data["wage"] > 0)

    def test_proportions_keys_and_arrays(self):
        self._check_dataset("proportions", {"temperature", "water", "germination_rate"})

    def test_proportions_size(self):
        data = load_dataset("proportions")
        assert data["germination_rate"].shape == (400,)

    def test_proportions_bounded(self):
        data = load_dataset("proportions")
        rate = data["germination_rate"]
        assert np.all(rate > 0) and np.all(rate < 1)

    def test_meuse_keys_and_arrays(self):
        self._check_dataset("meuse", {"x", "y", "dist", "zinc"})

    def test_meuse_size(self):
        data = load_dataset("meuse")
        assert data["zinc"].shape == (155,)

    def test_survival_keys_and_arrays(self):
        self._check_dataset("survival", {"time", "event", "age", "treatment"})

    def test_survival_size(self):
        data = load_dataset("survival")
        assert data["time"].shape == (250,)

    def test_survival_event_binary(self):
        data = load_dataset("survival")
        assert set(np.unique(data["event"])) <= {0.0, 1.0}

    def test_survival_time_positive(self):
        data = load_dataset("survival")
        assert np.all(data["time"] > 0)

    def test_abalone_keys_and_arrays(self):
        self._check_dataset(
            "abalone",
            {"length", "diameter", "height", "shucked_weight", "rings"},
        )

    def test_abalone_size(self):
        data = load_dataset("abalone")
        assert data["rings"].shape == (500,)

    def test_abalone_rings_positive(self):
        data = load_dataset("abalone")
        assert np.all(data["rings"] >= 1)

    def test_climate_keys_and_arrays(self):
        self._check_dataset("climate", {"month", "altitude", "latitude", "temperature"})

    def test_climate_size(self):
        data = load_dataset("climate")
        assert data["temperature"].shape == (600,)

    def test_climate_month_range(self):
        data = load_dataset("climate")
        assert np.all(data["month"] >= 1) and np.all(data["month"] <= 12)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def test_load_is_reproducible(self):
        data1 = load_dataset("mcycle")
        data2 = load_dataset("mcycle")
        np.testing.assert_array_equal(data1["times"], data2["times"])
        np.testing.assert_array_equal(data1["accel"], data2["accel"])


class TestLoadDatasetAsFrame:
    """load_dataset with as_frame=True returns a pandas DataFrame."""

    def test_as_frame_returns_dataframe(self):
        pd = pytest.importorskip("pandas")
        data = load_dataset("mcycle", as_frame=True)
        assert isinstance(data, pd.DataFrame)
        assert "times" in data.columns
        assert "accel" in data.columns
