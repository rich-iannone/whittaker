"""Built-in example datasets for cookbook examples and quick demonstrations.

All datasets are generated synthetically with fixed random seeds so they are fully reproducible
without any internet access or optional dependencies.  The statistical properties (family, effect
shapes, noise level) are designed to illustrate specific modeling scenarios.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict[str, str]] = {
    "mcycle": {
        "description": "Simulated motorcycle-crash accelerometer data (n=133).",
        "variables": "times (ms), accel (g)",
        "family": "Gaussian",
        "note": "Inspired by Silverman (1985). Useful for demonstrating heteroscedastic smoothing.",
    },
    "co2": {
        "description": "Synthetic monthly atmospheric CO2 concentrations (n=504, 1958–1999).",
        "variables": "t (decimal year), year, month, co2 (ppm)",
        "family": "Gaussian",
        "note": "Trend + annual seasonal cycle. Good for cyclic and additive smooths.",
    },
    "fish": {
        "description": "Synthetic fish-count survey data (n=300).",
        "variables": "temperature (°C), depth (m), count",
        "family": "Poisson",
        "note": (
            "Count response driven by a non-linear temperature effect and a linear depth effect."
        ),
    },
    "credit": {
        "description": "Synthetic credit-default dataset (n=1000).",
        "variables": "income (k$), debt_ratio, age, default (0/1)",
        "family": "Binomial",
        "note": "Binary outcome driven by smooth effects of income and debt ratio.",
    },
    "wages": {
        "description": "Synthetic worker-earnings dataset (n=800).",
        "variables": "age, experience (years), wage ($/hr)",
        "family": "Gamma",
        "note": "Log-wages shaped by smooth age and experience effects.",
    },
    "proportions": {
        "description": "Synthetic seed-germination dataset (n=400).",
        "variables": "temperature (°C), water (mm), germination_rate",
        "family": "Beta",
        "note": "Bounded [0, 1] response with a non-linear temperature optimum.",
    },
    "meuse": {
        "description": "Synthetic river-bank heavy-metals dataset (n=155).",
        "variables": "x, y (map coordinates), dist (to river), zinc (ppm)",
        "family": "Gaussian",
        "note": "Log(zinc) decreases with distance from the river. Good for spatial smooths.",
    },
    "survival": {
        "description": "Synthetic clinical-trial survival dataset (n=250).",
        "variables": "time (years), event (1=event), age, treatment (0/1)",
        "family": "CoxPH",
        "note": "Weibull hazard with a smooth age effect and a binary treatment arm.",
    },
    "abalone": {
        "description": "Synthetic abalone morphology dataset (n=500).",
        "variables": "length, diameter, height, shucked_weight, rings",
        "family": "Gaussian",
        "note": "Multiple predictors → ring count (proxy for age). Good for tensor products.",
    },
    "climate": {
        "description": "Synthetic climate station dataset (n=600).",
        "variables": "month, altitude (m), latitude, temperature (°C)",
        "family": "GaussianLS",
        "note": (
            "Individual temperature readings with heteroscedastic noise: variance increases "
            "with altitude and latitude.  Designed for GAMLSS location-scale modeling where "
            "both mean and dispersion are smooth functions of the predictors."
        ),
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_datasets() -> list[dict[str, str]]:
    """Return a list of all built-in datasets with their metadata.

    Returns
    -------
    list[dict[str, str]]
        Each entry has keys `name`, `description`, `variables`, `family`, and `note`.

    Examples
    --------
    ```{python}
    import whittaker as wk

    rows = wk.list_datasets()
    [r["name"] for r in rows]
    ```
    """
    return [{"name": k, **v} for k, v in _REGISTRY.items()]


def load_dataset(name: str, as_frame: bool = False) -> dict[str, NDArray] | Any:
    """Load a built-in example dataset.

    Parameters
    ----------
    name:
        Dataset name.  Call `list_datasets` to see all available names.
    as_frame:
        If `True`, return a `pandas.DataFrame` instead of a plain `dict`.
        Requires `pandas` to be installed.

    Returns
    -------
    dict[str, NDArray] or pandas.DataFrame
        Column-oriented data.  The `dict` form is accepted directly by
        `~whittaker.gam.GAM.fit` and all other Whittaker model classes.

    Examples
    --------
    ```{python}
    import whittaker as wk

    data = wk.load_dataset("mcycle")
    model = wk.GAM("accel ~ s(times)").fit(data)
    model.edf
    ```

    ```{python}
    df = wk.load_dataset("wages", as_frame=True)
    df.head()
    ```
    """
    _loaders: dict[str, Any] = {
        "mcycle": _mcycle,
        "co2": _co2,
        "fish": _fish,
        "credit": _credit,
        "wages": _wages,
        "proportions": _proportions,
        "meuse": _meuse,
        "survival": _survival,
        "abalone": _abalone,
        "climate": _climate,
    }
    if name not in _loaders:
        available = ", ".join(f"'{k}'" for k in _loaders)
        raise ValueError(f"Unknown dataset {name!r}. Available datasets: {available}.")
    data = _loaders[name]()
    if not as_frame:
        return data
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "load_dataset(..., as_frame=True) requires pandas. Install it with: pip install pandas"
        ) from exc
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def _mcycle() -> dict[str, NDArray]:
    """Simulated motorcycle-crash accelerometer data.

    Inspired by the classic dataset from Silverman (1985).  The response
    variable is head acceleration (g) measured at various times (ms) after
    simulated impact.  The signal is strongly non-linear and heteroscedastic,
    making it a standard stress-test for smoothing methods.
    """
    rng = np.random.default_rng(0)
    n = 133

    # Irregular time grid: more observations in the critical crash phase
    times = np.sort(
        np.concatenate(
            [
                rng.uniform(0.0, 13.0, 20),  # pre-impact
                rng.uniform(13.0, 32.0, 78),  # crash + recovery onset
                rng.uniform(32.0, 57.6, 35),  # late recovery
            ]
        )
    )
    times = np.round(times * 5) / 5  # quantise to 0.2 ms steps

    # Mean curve: near zero pre-impact, steep dip, gradual recovery
    def _mu(t: NDArray) -> NDArray:
        dip = -133.0 * np.exp(-0.5 * ((t - 19.0) / 5.5) ** 2)
        onset = 1.0 - np.exp(-np.maximum(t - 7.0, 0.0) / 4.0)
        return dip * onset

    mu = _mu(times)

    # Heteroscedastic noise: variance peaks at the crash
    sigma = 7.0 + 22.0 * np.exp(-0.5 * ((times - 19.0) / 10.0) ** 2)
    accel = mu + rng.normal(0.0, sigma, n)

    return {"times": times, "accel": accel}


def _co2() -> dict[str, NDArray]:
    """Synthetic monthly atmospheric CO2 concentrations (1958–1999).

    The signal has a rising linear trend (~1.6 ppm/year) and a dominant
    annual seasonal cycle (~6 ppm peak-to-trough), matching the broad
    character of the Mauna Loa record.
    """
    rng = np.random.default_rng(1)
    n = 504  # 42 years × 12 months

    year = np.repeat(np.arange(1958, 2000), 12)
    month = np.tile(np.arange(1, 13), 42)
    t = year + (month - 1) / 12.0  # decimal year

    trend = 315.0 + 1.65 * (t - 1958.0)
    seasonal = 3.2 * np.sin(2 * np.pi * (t - 1958.0 + 0.35))
    co2 = trend + seasonal + rng.normal(0.0, 0.4, n)

    return {
        "t": t,
        "year": year.astype(float),
        "month": month.astype(float),
        "co2": co2,
    }


def _fish() -> dict[str, NDArray]:
    """Synthetic fish-abundance survey data (Poisson counts).

    Fish counts are driven by a non-linear (hump-shaped) temperature
    effect peaking around 18 °C and a negative linear depth effect.
    """
    rng = np.random.default_rng(2)
    n = 300

    temperature = rng.uniform(8.0, 28.0, n)
    depth = rng.uniform(2.0, 50.0, n)

    log_mu = 2.8 - 0.018 * (temperature - 18.0) ** 2 - 0.025 * depth + rng.normal(0.0, 0.15, n)
    count = rng.poisson(np.exp(log_mu)).astype(float)

    return {"temperature": temperature, "depth": depth, "count": count}


def _credit() -> dict[str, NDArray]:
    """Synthetic credit-default dataset (binary response).

    Default probability follows a logistic model with non-linear effects
    of income and debt ratio.
    """
    rng = np.random.default_rng(3)
    n = 1000

    income = rng.uniform(20.0, 180.0, n)  # k$/yr, uniform scale
    debt_ratio = np.clip(rng.beta(2.0, 3.0, n), 0.0, 0.99)
    age = rng.uniform(20.0, 70.0, n)

    # Log-odds: higher debt ratio and lower income → higher default risk
    log_odds = (
        0.5
        - 0.018 * income
        + 4.5 * debt_ratio**2
        + 0.003 * (age - 45.0) ** 2 / 5.0
        + rng.normal(0.0, 0.25, n)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    default = rng.binomial(1, prob).astype(float)

    return {"income": income, "debt_ratio": debt_ratio, "age": age, "default": default}


def _wages() -> dict[str, NDArray]:
    """Synthetic worker-earnings dataset (Gamma-distributed wages).

    Log-wage is a smooth function of age (hump-shaped) and years of
    experience (concave), with multiplicative (Gamma) noise.
    """
    rng = np.random.default_rng(4)
    n = 800

    age = rng.uniform(18.0, 65.0, n)
    experience = np.clip(rng.exponential(scale=8.0, size=n), 0.0, age - 18.0)

    log_wage = (
        2.5
        + 0.06 * (age - 18.0)
        - 0.0006 * (age - 18.0) ** 2
        + 0.04 * experience
        - 0.0008 * experience**2
        + rng.normal(0.0, 0.25, n)
    )
    wage = np.exp(log_wage)

    return {"age": age, "experience": experience, "wage": wage}


def _proportions() -> dict[str, NDArray]:
    """Synthetic seed-germination dataset (Beta-distributed proportions).

    Germination rate has a non-linear optimum around 22 °C with respect
    to temperature, and increases with water availability up to saturation.
    """
    rng = np.random.default_rng(5)
    n = 400

    temperature = rng.uniform(5.0, 38.0, n)
    water = rng.uniform(0.0, 120.0, n)

    # Mean germination on logit scale
    logit_mu = (
        2.0
        - 0.012 * (temperature - 22.0) ** 2
        + 0.03 * water
        - 0.0002 * water**2
        + rng.normal(0.0, 0.5, n)
    )
    mu = 1.0 / (1.0 + np.exp(-logit_mu))
    # Beta-distributed outcome
    phi = 15.0  # concentration
    a = mu * phi
    b = (1.0 - mu) * phi
    germination_rate = rng.beta(np.maximum(a, 0.01), np.maximum(b, 0.01))
    germination_rate = np.clip(germination_rate, 1e-4, 1.0 - 1e-4)

    return {"temperature": temperature, "water": water, "germination_rate": germination_rate}


def _meuse() -> dict[str, NDArray]:
    """Synthetic river-bank heavy-metals dataset (spatial).

    Zinc concentration (ppm) decreases with distance from the river.
    The (x, y) coordinates lie on a stylised floodplain and `dist`
    is the normalised distance to the river bank (0 = bank, 1 = far).
    """
    rng = np.random.default_rng(6)
    n = 155

    # River runs roughly NW–SE; sample points on the floodplain
    river_t = rng.uniform(0.0, 10.0, n)
    perp = rng.exponential(scale=0.8, size=n)  # distance perpendicular to river
    x = 178500.0 + 60.0 * river_t - 20.0 * perp + rng.normal(0, 5, n)
    y = 329700.0 + 30.0 * river_t + 50.0 * perp + rng.normal(0, 5, n)
    dist = np.clip(perp / perp.max(), 0.01, 1.0)

    # Log(zinc) decreases with distance from the river
    log_zinc = 6.9 - 1.8 * dist + 0.4 * np.sin(river_t / 2.0) + rng.normal(0.0, 0.3, n)
    zinc = np.exp(log_zinc)

    return {"x": x, "y": y, "dist": dist, "zinc": zinc}


def _survival() -> dict[str, NDArray]:
    """Synthetic clinical-trial survival dataset (Cox PH).

    Event times follow a Weibull baseline with a smooth age effect and a
    protective binary treatment effect.  Approximately 30 % of observations
    are administratively censored at 5 years.
    """
    rng = np.random.default_rng(7)
    n = 250

    age = rng.uniform(30.0, 75.0, n)
    treatment = rng.binomial(1, 0.5, n).astype(float)

    # Log-hazard relative to baseline
    log_hr = 0.025 * (age - 55.0) - 0.8 * treatment + rng.normal(0.0, 0.2, n)
    hr = np.exp(log_hr)

    # Weibull baseline: shape=1.5, scale=4 years
    shape, scale = 1.5, 4.0
    # Inverse CDF of Weibull proportional-hazards model
    u = rng.uniform(0.0, 1.0, n)
    time_event = scale * (-np.log(u) / hr) ** (1.0 / shape)

    # Administrative censoring at 5 years
    censor_time = 5.0
    time = np.minimum(time_event, censor_time)
    status = (time_event <= censor_time).astype(float)

    return {"time": time, "event": status, "age": age, "treatment": treatment}


def _abalone() -> dict[str, NDArray]:
    """Synthetic abalone morphology dataset.

    Physical measurements of abalones (length, diameter, height, shucked
    weight) are used to predict ring count, which is a proxy for age.
    The true ring count is a smooth non-linear function of the measurements
    with moderate noise.
    """
    rng = np.random.default_rng(8)
    n = 500

    length = rng.uniform(0.10, 0.80, n)
    diameter = length * rng.uniform(0.65, 0.85, n)
    height = length * rng.uniform(0.10, 0.40, n)
    shucked_weight = rng.uniform(0.002, 1.5, n) * length**2.5

    log_rings = (
        1.6
        + 2.5 * length
        - 1.2 * length**2
        + 0.4 * np.log(shucked_weight + 0.1)
        + rng.normal(0.0, 0.18, n)
    )
    rings = np.round(np.clip(np.exp(log_rings), 1.0, 29.0))

    return {
        "length": length,
        "diameter": diameter,
        "height": height,
        "shucked_weight": shucked_weight,
        "rings": rings,
    }


def _climate() -> dict[str, NDArray]:
    """Synthetic climate-station dataset (GAMLSS / location-scale).

    Monthly mean temperature and its station-level standard deviation both
    vary with altitude and latitude.  Designed for GAMLSS models where the
    scale parameter also needs a smooth predictor.
    """
    rng = np.random.default_rng(9)
    n = 600

    altitude = rng.uniform(0.0, 3000.0, n)
    latitude = rng.uniform(40.0, 70.0, n)
    month = rng.integers(1, 13, size=n).astype(float)

    # True mean temperature (varies with altitude, latitude, and season)
    mu = (
        25.0
        - 0.006 * altitude
        - 0.5 * (latitude - 40.0)
        + 8.0 * np.cos(2 * np.pi * (month - 1) / 12.0)
    )
    # True SD: larger at high altitude and high latitude (more variable climates)
    log_sd = 0.8 + 0.0004 * altitude + 0.025 * (latitude - 40.0)
    sigma = np.exp(log_sd)
    temperature = mu + rng.normal(0.0, sigma, n)

    return {
        "month": month,
        "altitude": altitude,
        "latitude": latitude,
        "temperature": temperature,
    }
