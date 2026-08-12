r"""Abstract base class for GAMLSS distributional families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class GAMLSSFamily(ABC):
    r"""Abstract base class for GAMLSS distributional families.

    Ordinary `Family` subclasses (used by `GAM`) model only the mean `mu` of the response as a
    function of covariates, treating any other distributional parameters (e.g. the variance or
    dispersion) as constant across observations. A `GAMLSSFamily`, used by `GAMLSS` instead of
    `GAM`, generalizes the single-parameter `Family` abstraction to the location-scale-shape
    setting: instead of one response parameter with a mean link, it defines a full response
    distribution with `K` named parameters $\theta_1, \ldots, \theta_K$ (e.g. location `mu` and
    scale `sigma`), each with its own link function $g_k$ and its own additive predictor
    $\eta_k = g_k(\theta_k)$. This is useful whenever more than the mean of the response changes
    systematically with covariates — for example, when the spread (heteroscedasticity), skew, or
    zero-inflation probability also varies across the range of the predictors.

    `GAMLSS` fits every parameter's additive predictor jointly by alternating penalized IRLS
    updates across parameters (the RS algorithm), which relies on each family supplying the
    per-parameter score `dl_dtheta`, (expected) Fisher information `d2l_dtheta2`,
    link/inverse-link/link-derivative, log-likelihood, initial values, and a `simulate` method.
    Subclasses must implement all of the abstract methods below.

    Whittaker ships with the following concrete GAMLSS families:

    - `GaussianLS` — location-scale Gaussian: identity link for the mean, log link for the
      standard deviation.
    - `GammaLS` — location-scale Gamma: log link for both the mean and the coefficient of
      variation.
    - `BetaLS` — mean-precision Beta: logit link for the mean, log link for the precision.
    - `ZeroInflatedPoisson` — Poisson mean plus a zero-inflation probability, for count data
      with excess zeros.
    - `ZeroInflatedNegativeBinomial` — overdispersed counts with excess zeros, combining
      `NegativeBinomial`-style overdispersion with zero-inflation.

    Examples
    --------
    GAMLSS families are passed to `GAMLSS`, and each distributional parameter gets its own
    formula:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.sin(x)
    sigma = 0.2 + 0.3 * np.abs(np.cos(x))
    y = rng.normal(mu, sigma)

    data = {"x": x, "y": y}

    # Model both the mean and the standard deviation as smooth functions of x
    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "sigma": "y ~ s(x)"},
        family=wk.GaussianLS(),
    )
    model.fit(data)
    print(model.summary())
    ```
    """

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Names of distributional parameters, e.g. `('mu', 'sigma')`."""
        ...

    @abstractmethod
    def link(self, param: str, values: NDArray) -> NDArray:
        """Apply the link function for `param`: `eta = g(theta)`."""
        ...

    @abstractmethod
    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        """Apply the inverse link for `param`: `theta = g^-1(eta)`."""
        ...

    @abstractmethod
    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        """Derivative of the link for `param`: `d(eta)/d(theta)`."""
        ...

    @abstractmethod
    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """First derivative of the log-likelihood with respect to `param`."""
        ...

    @abstractmethod
    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """Negative expected second derivative of the log-likelihood with respect to `param`.

        Must return positive values (the expected Fisher information diagonal for this
        parameter), since it is used directly as a working weight during fitting.
        """
        ...

    @abstractmethod
    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        """Full log-likelihood evaluated at the given parameter values."""
        ...

    @abstractmethod
    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Starting values for all distributional parameters given `y`."""
        ...

    @abstractmethod
    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        """Simulate response values from the distribution."""
        ...
