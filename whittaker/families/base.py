r"""Abstract base class for response distribution families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from numpy.typing import NDArray

if TYPE_CHECKING:
    import numpy as np


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
        r"""Apply the link function $g(\mu)$, mapping the conditional mean to the linear predictor.

        The link function defines the relationship $\eta = g(\mu)$ between the mean of the
        response and the linear predictor. Each family provides a canonical or default link;
        for example, the identity link for Gaussian, the log link for Poisson, and the logit
        link for Binomial.

        Parameters
        ----------
        mu
            Conditional mean values $\mu$, shape `(n,)`. Must lie in the valid range for
            the family (e.g., $\mu > 0$ for Poisson, $0 < \mu < 1$ for Binomial).

        Returns
        -------
        NDArray
            Linear predictor values $\eta = g(\mu)$, shape `(n,)`.
        """
        ...

    @abstractmethod
    def link_inverse(self, eta: NDArray) -> NDArray:
        r"""Apply the inverse link $g^{-1}(\eta)$, mapping the linear predictor back to the mean.

        This is the transformation applied to the fitted linear predictor to recover fitted
        values `mu` on the response scale, e.g. after `GAM.predict()` computes `eta`.

        Parameters
        ----------
        eta
            Linear predictor values $\eta$, shape `(n,)`.

        Returns
        -------
        NDArray
            Conditional mean values $\mu = g^{-1}(\eta)$, shape `(n,)`.
        """
        ...

    @abstractmethod
    def link_derivative(self, mu: NDArray) -> NDArray:
        r"""Derivative of the link function, $g'(\mu) = d\eta/d\mu$.

        Used by the P-IRLS fitting loop to form the working response `z` and working weights
        `W` at each iteration, via a first-order (delta-method) linearization of the link
        function around the current fit.

        Parameters
        ----------
        mu
            Conditional mean values $\mu$, shape `(n,)`.

        Returns
        -------
        NDArray
            Derivative values $g'(\mu)$, shape `(n,)`.
        """
        ...

    @abstractmethod
    def variance(self, mu: NDArray) -> NDArray:
        r"""Variance function $V(\mu)$ relating the response variance to its mean.

        The conditional variance of the response is $\operatorname{Var}(Y) = \phi \, V(\mu)$,
        where $\phi$ is the dispersion (scale) parameter. For the Gaussian family this is
        constant (`1`); for Poisson it is `mu`; for Gamma it is `mu^2`, etc. `variance` is used
        by P-IRLS to form the working weights.

        Parameters
        ----------
        mu
            Conditional mean values $\mu$, shape `(n,)`.

        Returns
        -------
        NDArray
            Variance function values $V(\mu)$, shape `(n,)`.
        """
        ...

    @abstractmethod
    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        r"""Total (unscaled) deviance.

        $D(y, \hat\mu) = 2 \sum_i [\ell(y_i; y_i) - \ell(y_i; \hat\mu_i)]$.

        The deviance measures the discrepancy between the fitted model and a saturated model
        that fits the data exactly. It is used by `GAM.fit()` for smoothing parameter selection
        (GCV) and by `GAM.summary()` for goodness-of-fit reporting.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        weights
            Optional prior weights, shape `(n,)`. When given, the deviance is
            $\sum_i w_i d_i$ rather than $\sum_i d_i$.

        Returns
        -------
        float
            The total deviance.
        """
        ...

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        r"""Per-observation deviance contributions $d_i$, before summing over observations.

        The total deviance returned by `deviance` is $\sum_i d_i$ (or the weighted sum). The
        default implementation here is `(y - mu)^2`, appropriate for the Gaussian family;
        concrete families with a different deviance formula override this method.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.

        Returns
        -------
        NDArray
            Per-observation deviance contributions $d_i$, shape `(n,)`.
        """
        return (y - mu) ** 2

    @abstractmethod
    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        r"""Log-likelihood $\ell(y; \mu, \phi)$ evaluated at the given scale parameter $\phi$.

        Used by `GAM.fit()` for REML/ML-based smoothing parameter selection and by
        `GAM.summary()` for reporting AIC and related fit statistics.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        scale
            Dispersion (scale) parameter $\phi$.
        weights
            Optional prior weights, shape `(n,)`. When given, the log-likelihood is
            $\sum_i w_i \ell_i$ rather than $\sum_i \ell_i$.

        Returns
        -------
        float
            The total log-likelihood.
        """
        ...

    @property
    def scale_known(self) -> bool:
        """Whether the dispersion (scale) parameter is fixed rather than estimated.

        Returns `True` for families with no free dispersion parameter (e.g. `Binomial`,
        `Poisson`), in which case the scale is always `1` and is not estimated during fitting.
        Returns `False` (the default) for families such as `Gaussian` and `Gamma`, whose scale
        is estimated from the data.
        """
        return False

    @abstractmethod
    def simulate(self, mu: NDArray, scale: float, rng: np.random.Generator) -> NDArray:
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

    def log_lik_pointwise(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> NDArray:
        r"""Per-observation log-likelihood contributions $\ell_i(y_i; \mu_i, \phi)$.

        Returns the same values that `log_likelihood` sums, without summing, as a 1-D array. Used by
        `GAM.loo()` to compute PSIS-LOO cross-validation.

        The default implementation calls `log_likelihood` on each observation individually. Key
        families override this with a vectorized implementation.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Fitted conditional mean values, shape `(n,)`.
        scale
            Dispersion (scale) parameter $\phi$.
        weights
            Optional prior weights, shape `(n,)`.

        Returns
        -------
        NDArray
            Per-observation log-likelihood values, shape `(n,)`.
        """
        import numpy as np

        n = len(y)
        out = np.empty(n)
        for i in range(n):
            w_i = None if weights is None else weights[i : i + 1]
            out[i] = self.log_likelihood(y[i : i + 1], mu[i : i + 1], scale, weights=w_i)
        return out

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray] | None:
        """Custom IRLS pseudo-response and working weights.

        Override in families whose loss does not fit the standard GLM deviance framework (e.g.
        `QuantileFamily`, `CoxPH`, `OrderedCategorical`, `Multinomial`), to supply the working
        response `z` and working weights `W` directly rather than deriving them from `link`,
        `link_derivative`, and `variance`. Returning `None` (the default) tells the P-IRLS loop
        to fall back to the standard GLM formula
        `z = eta + (y - mu) * link_derivative(mu)`,
        `W = 1 / (link_derivative(mu)^2 * variance(mu))`.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.
        mu
            Current fitted mean values, shape `(n,)`.
        eta
            Current linear predictor values, shape `(n,)`.

        Returns
        -------
        tuple of NDArray, or None
            `(z, W)`, the pseudo-response and diagonal working-weight vector, each shape `(n,)`,
            or `None` to use the default GLM formula.
        """
        return None

    def initialize(self, y: NDArray) -> NDArray:
        """Compute starting values for `mu` given the observed response `y`.

        Called once before the first P-IRLS iteration to seed the mean. The default
        implementation returns `y` unchanged, which is appropriate for the `Gaussian` family
        with its identity link. Families with non-identity links or constrained means (e.g.
        `Poisson`, `Binomial`, `Gamma`) override this to nudge `y` into the valid range and
        avoid numerical issues (such as `log(0)`) on the first iteration.

        Parameters
        ----------
        y
            Observed response values, shape `(n,)`.

        Returns
        -------
        NDArray
            Starting values for `mu`, shape `(n,)`.
        """
        return y.copy()
