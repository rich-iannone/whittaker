r"""Abstract base class for response distribution families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Family(ABC):
    r"""Abstract family defining the response distribution and link function.

    A `Family` encapsulates everything the P-IRLS fitting loop needs to know about the
    conditional distribution of the response `y` given the linear predictor `eta`. Every GLM
    and GAM family in Whittaker belongs to the exponential dispersion family, and is fully
    characterized by three ingredients:

    1. The **link function** `g`, relating the mean `mu` to the linear predictor
       `eta`: `eta = g(mu)`, along with its inverse and derivative.
    2. The **variance function** `V(mu)`, relating the variance of the response to its mean:
       `Var(Y) = phi * V(mu)`, where `phi` is the dispersion (scale) parameter.
    3. The **deviance** and **log-likelihood**, which quantify goodness of fit and are used by
       `GAM.fit()` for smoothing parameter selection (GCV/REML) and by `GAM.summary()` for
       reporting.

    Subclasses must implement all abstract methods (`link`, `link_inverse`, `link_derivative`,
    `variance`, `deviance`, `log_likelihood`, `simulate`). The link function and its
    inverse/derivative are used by the P-IRLS algorithm to form pseudo-data (the working
    response `z`) and working weights `W` at each iteration. Families whose loss does not fit
    the standard GLM deviance framework (e.g. `QuantileFamily`, `CoxPH`, `OrderedCategorical`,
    `Multinomial`) may instead override `irls_update` to supply `z` and `W` directly.

    Whittaker ships with the following concrete families:

    - `Gaussian` — identity link, constant variance; the default family for continuous,
      unbounded responses.
    - `Poisson` — log link, `V(mu) = mu`; for count data.
    - `Binomial` — logit link, `V(mu) = mu(1-mu)`; for binary or proportion responses.
    - `Gamma` — log link, `V(mu) = mu^2`; for positive, right-skewed continuous data.
    - `NegativeBinomial` — log link, `V(mu) = mu + mu^2/theta`; for overdispersed counts.
    - `Beta` — logit link; for proportions strictly between 0 and 1.
    - `Tweedie` / `TweedieEstimated` (via `tw()`) — log link, `V(mu) = mu^p`; for compound
      Poisson-Gamma data with a point mass at zero (e.g. insurance claims).
    - `InverseGaussian` — log link, `V(mu) = mu^3`; for positive, heavy-tailed continuous data.
    - `CoxPH` — proportional hazards partial likelihood; for survival/time-to-event data.
    - `OrderedCategorical` — cumulative logit (proportional odds); for ordinal responses.
    - `Multinomial` — baseline-category logit; for unordered categorical responses.
    - `QuantileFamily` — Extended Log-F (ELF) smooth pinball loss; for quantile regression.

    For distributional regression, where more than one parameter of the response distribution
    (e.g. both the mean and the scale) is modeled by its own smooth predictor, see
    `GAMLSSFamily` and its concrete subclasses (`GaussianLS`, `GammaLS`, `BetaLS`,
    `ZeroInflatedPoisson`, `ZeroInflatedNegativeBinomial`).

    Examples
    --------
    Families are passed to `GAM` via the `family` argument; they are rarely instantiated by
    users beyond that.

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0, 2 * np.pi, n)
    y = rng.poisson(np.exp(0.5 * np.sin(x)))

    data = {"x": x, "y": y}

    # Any concrete Family subclass can be passed to GAM
    model = wk.GAM("y ~ s(x)", family=wk.Poisson())
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    @abstractmethod
    def link(self, mu: NDArray) -> NDArray:
        """Apply the link function: η = g(μ)."""
        ...

    @abstractmethod
    def link_inverse(self, eta: NDArray) -> NDArray:
        """Apply the inverse link: μ = g⁻¹(η)."""
        ...

    @abstractmethod
    def link_derivative(self, mu: NDArray) -> NDArray:
        """Derivative of the link function: dη/dμ = g'(μ)."""
        ...

    @abstractmethod
    def variance(self, mu: NDArray) -> NDArray:
        """Variance function V(μ).

        For the Gaussian family this is constant (1); for Poisson it is μ, etc.
        """
        ...

    @abstractmethod
    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        """Total (unscaled) deviance: 2 * Σ [ℓ(y; y) − ℓ(y; μ)].

        When *weights* is given the deviance is Σ w_i d_i.
        """
        ...

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        """Per-observation deviance contributions d_i (before summing).

        Default implementation: `(y - mu)^2` (Gaussian). Override for other families.
        """
        return (y - mu) ** 2

    @abstractmethod
    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        """Log-likelihood ℓ(y; μ, φ) evaluated at the given scale parameter φ.

        When *weights* is given the log-likelihood is Σ w_i ℓ_i.
        """
        ...

    @property
    def scale_known(self) -> bool:
        """Whether the scale parameter is fixed (True for Binomial, Poisson)."""
        return False

    @abstractmethod
    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        """Simulate response values from the distribution.

        Parameters
        ----------
        mu:
            Mean (fitted values), shape `(n,)`.
        scale:
            Estimated scale parameter φ.
        rng:
            A `numpy.random.Generator` instance.

        Returns
        -------
        NDArray
            Simulated response values, shape `(n,)`.
        """
        ...

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray] | None:
        """Custom IRLS pseudo-response and working weights.

        Override in families whose loss is not a standard GLM deviance (e.g. quantile regression).
        Return `(z, W)` where *z* is the pseudo-response and *W* is the diagonal working-weight
        vector. Return `None` to use the default GLM formula.
        """
        return None

    def initialize(self, y: NDArray) -> NDArray:
        """Starting values for μ given the response *y*.

        The default returns *y* unchanged, which is appropriate for Gaussian with identity link.
        Families with non-identity links or constrained means should override this.
        """
        return y.copy()
