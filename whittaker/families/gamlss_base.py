r"""Abstract base class for GAMLSS distributional families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    import numpy as np


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
        """Names of the distributional parameters modeled by this family.

        Every concrete `GAMLSSFamily` names its parameters explicitly, e.g.
        `('mu', 'sigma')` for `GaussianLS` and `GammaLS`, or `('mu', 'phi')` for `BetaLS`.
        `GAMLSS` uses this tuple to determine which formulas it expects (one per name) and
        the order in which it cycles through parameters during the RS fitting algorithm.

        Returns
        -------
        tuple[str, ...]
            The parameter names, in the order they should be updated during fitting.
        """
        ...

    @abstractmethod
    def link(self, param: str, values: NDArray) -> NDArray:
        r"""Apply the link function for `param`: `eta = g(theta)`.

        Maps a distributional parameter from its natural scale (e.g. `mu` on the real
        line, `sigma > 0`) onto the unconstrained scale of the additive predictor `eta`,
        so that `GAMLSS` can model it with an ordinary smooth linear predictor.

        Parameters
        ----------
        param
            Name of the distributional parameter being linked, one of `parameter_names`.
        values
            Parameter values on the natural scale, shape `(n,)`.

        Returns
        -------
        NDArray
            Linked values `eta = g(theta)`, shape `(n,)`.
        """
        ...

    @abstractmethod
    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        r"""Apply the inverse link for `param`: `theta = g^-1(eta)`.

        Maps the additive predictor back onto the natural scale of the parameter,
        enforcing any constraints (e.g. positivity for a scale parameter, or the unit
        interval for a probability) implied by the chosen link.

        Parameters
        ----------
        param
            Name of the distributional parameter being linked, one of `parameter_names`.
        eta
            Additive predictor values, shape `(n,)`.

        Returns
        -------
        NDArray
            Parameter values on the natural scale, shape `(n,)`.
        """
        ...

    @abstractmethod
    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        r"""Derivative of the link for `param`: `d(eta)/d(theta)`.

        Used by the RS algorithm to convert the score `dl_dtheta` into a working
        pseudo-response on the scale of the additive predictor, via the chain rule
        $\partial \ell / \partial \eta = (\partial \ell / \partial \theta)\, (d\eta/d\theta)^{-1}$.

        Parameters
        ----------
        param
            Name of the distributional parameter being linked, one of `parameter_names`.
        values
            Parameter values on the natural scale at which to evaluate the derivative,
            shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise derivatives `d(eta)/d(theta)`, shape `(n,)`.
        """
        ...

    @abstractmethod
    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""First derivative of the log-likelihood with respect to `param`.

        Returns $\partial \ell / \partial \theta$ evaluated elementwise at the current
        parameter estimates. This score forms the basis of the working pseudo-response
        used by `GAMLSS`'s RS algorithm when updating the additive predictor for `param`,
        holding the other distributional parameters fixed.

        Parameters
        ----------
        param
            Name of the distributional parameter to differentiate with respect to, one of
            `parameter_names`.
        y
            Observed response values, shape `(n,)`.
        params
            Current estimates of all distributional parameters, keyed by name, each of
            shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise first derivatives, shape `(n,)`.
        """
        ...

    @abstractmethod
    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Negative expected second derivative of the log-likelihood with respect to `param`.

        Returns the (expected) Fisher information $-E[\partial^2 \ell / \partial \theta^2]$
        for `param`, evaluated elementwise at the current parameter estimates. Must return
        positive values, since it is used directly as a working weight during fitting: it
        both scales the working pseudo-response and forms the diagonal weight matrix for the
        penalized IRLS update of `param`'s additive predictor.

        Parameters
        ----------
        param
            Name of the distributional parameter to differentiate with respect to, one of
            `parameter_names`.
        y
            Observed response values, shape `(n,)`.
        params
            Current estimates of all distributional parameters, keyed by name, each of
            shape `(n,)`.

        Returns
        -------
        NDArray
            Elementwise working weights, shape `(n,)`, guaranteed positive.
        """
        ...

    @abstractmethod
    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        """Full log-likelihood evaluated at the given parameter values.

        Sums the per-observation log-density of the response distribution across all
        observations, using the current estimate of every distributional parameter. Used
        for convergence checks during fitting and for computing information criteria such
        as AIC/BIC in model summaries.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        params
            Current estimates of all distributional parameters, keyed by name, each of
            shape `(n,)`.

        Returns
        -------
        float
            The total log-likelihood summed over all observations.
        """
        ...

    @abstractmethod
    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Starting values for all distributional parameters given `y`.

        Produces a reasonable initial guess for every parameter in `parameter_names` from
        the raw response alone, before any covariate information is used. `GAMLSS` uses
        these as the starting point for the first RS iteration.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.

        Returns
        -------
        dict[str, NDArray]
            Initial values for each parameter, keyed by name, each of shape `(n,)`.
        """
        ...

    @abstractmethod
    def simulate(self, params: dict[str, NDArray], rng: np.random.Generator) -> NDArray:
        """Simulate response values from the distribution.

        Draws one random sample per observation from the response distribution at the
        given per-observation parameter values. Used for posterior/parametric-bootstrap
        style simulation from a fitted `GAMLSS` model.

        Parameters
        ----------
        params
            Distributional parameter values to simulate from, keyed by name, each of
            shape `(n,)`.
        rng
            A NumPy random generator (e.g. `numpy.random.default_rng()`) used to draw the
            simulated values.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        ...
