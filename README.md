<p align="center">
<a href="https://rich-iannone.github.io/whittaker/">
<img src="https://rich-iannone.github.io/whittaker/assets/whittaker_logo.png" alt="Whittaker" width="350">
</a>
</p>
<p align="center">Next-generation GAMs for Python: flexible smoothing, principled inference, beautiful output.</p>
<p align="center">
<a href="https://pypi.org/project/whittaker/"><img src="https://img.shields.io/pypi/v/whittaker?logo=python&logoColor=white&color=orange" alt="PyPI"></a>
<a href="https://pypi.org/project/whittaker/"><img src="https://img.shields.io/pypi/pyversions/whittaker.svg" alt="Python versions"></a>
<a href="https://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
<a href="https://github.com/rich-iannone/whittaker/actions/workflows/ci.yml"><img src="https://github.com/rich-iannone/whittaker/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
</p>
<p align="center">
<a href="https://www.repostatus.org/#wip"><img src="https://www.repostatus.org/badges/latest/wip.svg" alt="Repo Status"></a>
</p>
<p align="center">
<a href="https://rich-iannone.github.io/whittaker/"><img src="https://img.shields.io/badge/docs-project_website-blue.svg" alt="Documentation"></a>
<a href="https://github.com/rich-iannone/whittaker/graphs/contributors"><img src="https://img.shields.io/github/contributors/rich-iannone/whittaker" alt="Contributors"></a>
<a href="https://www.contributor-covenant.org/version/3/0/"><img src="https://img.shields.io/badge/Contributor%20Covenant-v3.0%20adopted-ff69b4.svg" alt="Contributor Covenant"></a>
</p>

---

## What is Whittaker?

Whittaker is a Python library for Generalized Additive Models (GAMs), the flexible regression models that replace rigid linear assumptions with smooth, data-driven functions. Whether you're fitting a dose-response curve, modeling spatial patterns, or building prediction intervals, Whittaker gives you the smooth catalog, inference machinery, and diagnostic tools to do it right.

### Why Whittaker?

- it works with your dataframe library: Pandas, Polars, PyArrow, or anything supported by [Narwhals](https://narwhals-dev.github.io/narwhals/)
- it gives you the full smooth catalog from R's mgcv: thin plate regression splines, cubic splines, P-splines, tensor products, cyclic splines, random effects, factor smooths, and more
- smoothness selection is principled: REML by default, with GCV, ML, and fREML as alternatives
- you get beautiful, interactive visualizations with [Altair](https://altair-viz.github.io/) and publication-ready tables through [Great Tables](https://posit-co.github.io/great-tables/)
- it goes beyond the mean: distributional regression (GAMLSS), quantile regression, conformal prediction, causal inference, streaming GAMs, and functional regression are all built in

## What's included

Core GAM fitting:

- **GAM**: the central class. Fit penalized regression splines with automatic smoothness selection via REML, GCV, or ML. Full summary, diagnostics, and partial-effect visualization.
- **Formula syntax**: R-style formulas like `"y ~ s(x1) + s(x2, k=20) + te(x3, x4) + x5"` with smooth terms, tensor products, linear terms, interactions, offsets, and by-variable smooths.
- **Response families**: Gaussian, Poisson, Binomial, Gamma, Negative Binomial, Beta, Tweedie, Inverse Gaussian, Cox PH, and more. Each with appropriate link functions and variance structure.
- **Smooth basis types**: TPRS (default), cubic regression splines, P-splines, cyclic variants, shrinkage smooths, thin plate splines, Duchon splines, Gaussian processes, soap film smooths, Markov random fields, random effects, and factor smooths.
- **Shape constraints**: monotone increasing/decreasing, convex, and concave smooths via constrained P-splines with PAVA projection.

Prediction and inference:

- **Prediction**: point estimates, standard errors, confidence intervals (pointwise and simultaneous), prediction intervals, and term-level contributions. All on response or link scale.
- **Diagnostics**: `model.summary()` for EDF and significance tests, `model.check()` for basis dimension adequacy (k-index test), concurvity analysis, and residual plots.
- **Cross-validation**: k-fold CV with deviance, MSE, or MAE scoring via `cross_validate()`.

Advanced models:

- **Distributional regression** (`GAMLSS`): model location, scale, and shape simultaneously. Gaussian, Gamma, and Beta location-scale families, plus zero-inflated Poisson and Negative Binomial.
- **Quantile regression** (`QuantileGAM`): fit conditional quantiles with ELF loss, optional non-crossing constraints, and sigma calibration.
- **Conformal prediction** (`ConformalPredictor`): distribution-free prediction intervals via split, CV+, and jackknife+ methods.
- **Causal inference** (`CausalGAM`): double/debiased machine learning for ATE and CATE estimation, with mediation analysis.
- **Streaming GAMs** (`StreamingGAM`): incremental fitting via sufficient statistics with exponential decay for tracking distribution shift.
- **Multi-response GAMs** (`MultiResponseGAM`): joint fitting of multiple responses with optional residual correlation modeling.
- **Functional regression** (`FunctionalGAM`): scalar-on-function regression with B-spline or Fourier bases for functional covariates.

Scalability and deployment:

- **Large datasets**: `BigGAM` (discretized P-IRLS), `PolarsGAM` (streaming from Polars/files), and `DuckDBGAM` (SQL-native streaming) for datasets that exceed memory.
- **Serialization**: `save_gam` / `load_gam` for compact `.npz` archives, and `to_mgcv_dict` / `from_mgcv_dict` for R interoperability.
- **scikit-learn integration**: `GAMRegressor` and `GAMClassifier` for use in pipelines and grid search.

## Get started

Here's a simple example: fit a smooth to noisy data, inspect the fit, and predict on new observations.

```python
import numpy as np
import whittaker as wk

# Generate some data
rng = np.random.default_rng(23)
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x) + rng.normal(0, 0.3, 200)

# Fit a GAM with automatic smoothness selection
model = wk.GAM("y ~ s(x)")
model.fit({"x": x, "y": y}, method="REML")
model.summary()
```

```python
# Predict with standard errors
preds = model.predict({"x": np.linspace(0, 2 * np.pi, 50)}, se=True)
```

<p align="center">
<img src="https://raw.githubusercontent.com/rich-iannone/whittaker/main/assets/readme_partial_effects.png" alt="Partial effects plot" width="700">
</p>

```python
# Check model adequacy
model.check()
```

<p align="center">
<img src="https://raw.githubusercontent.com/rich-iannone/whittaker/main/assets/readme_diagnostics.png" alt="Diagnostic plots" width="700">
</p>

See the [user guide](https://rich-iannone.github.io/whittaker/user-guide/get-started.html) for a comprehensive tour of all features, from basic smoothing to distributional regression and causal inference.

## See more

A more complete example showing multiple predictors, a non-Gaussian family, and model comparison:

```python
import numpy as np
import whittaker as wk

# Simulate Poisson count data with two smooth effects
rng = np.random.default_rng(23)
n = 500
x1 = rng.uniform(0, 2 * np.pi, n)
x2 = rng.uniform(0, 1, n)
mu = np.exp(0.5 + 0.8 * np.sin(x1) + 2 * x2)
y = rng.poisson(mu).astype(float)

data = {"x1": x1, "x2": x2, "y": y}

# Fit a Poisson GAM
model = wk.GAM("y ~ s(x1) + s(x2)", family=wk.Poisson())
model.fit(data, method="REML")
model.summary()

# Cross-validate to check out-of-sample performance
cv = wk.cross_validate("y ~ s(x1) + s(x2)", data, family=wk.Poisson(), n_folds=5)
print(f"CV score: {cv.cv_score:.4f} (SE: {cv.cv_se:.4f})")

# Predict on new data with confidence intervals
preds = model.predict(
    {"x1": np.linspace(0, 2 * np.pi, 100), "x2": np.full(100, 0.5)},
    interval="confidence", level=0.95,
)

# Save the fitted model for deployment
wk.save_gam(model, "poisson_model.npz")
loaded = wk.load_gam("poisson_model.npz")
```

## Installation

```bash
pip install whittaker
```

For optional backends and visualization:

```bash
pip install "whittaker[all]"       # Pandas, Polars, PyArrow, Altair, Great Tables
pip install "whittaker[pl,altair]" # Just Polars and Altair
```

## License

MIT (c) Richard Iannone.
