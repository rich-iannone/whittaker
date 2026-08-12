r"""Zero-inflated Poisson (ZIP) and zero-inflated negative binomial (ZINB) GAMLSS families."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, gammaln, logit

from whittaker.families.gamlss_base import GAMLSSFamily

_EPS = np.finfo(float).eps
_MU_FLOOR = 1e-10


class ZeroInflatedPoisson(GAMLSSFamily):
    r"""Zero-inflated Poisson (ZIP) family for GAMLSS.

    Count data often has more zeros than a plain `Poisson` model can explain — for example,
    when some observations are structurally incapable of the event occurring at all (a
    "never-taker" always reports zero), in addition to the zeros that arise simply because the
    Poisson mean is low. `ZeroInflatedPoisson` models this as a mixture: with probability `pi`
    an observation is a structural zero, and with probability `1 - pi` it is drawn from an
    ordinary `Poisson(mu)` distribution (which can itself still produce a zero). Both `mu` and
    `pi` are modeled as smooth functions of covariates through `GAMLSS`, so the excess-zero
    probability and the count intensity can each vary independently across the covariate space.
    If overdispersion remains even among the non-structural-zero counts, use
    `ZeroInflatedNegativeBinomial` instead.

    Parameters
    ----------
    None
        `ZeroInflatedPoisson` takes no constructor arguments; both `mu` and `pi` are modeled
        entirely through the formulas supplied to `GAMLSS`.

    Notes
    -----
    Two distributional parameters are modeled, each with its own link:

    $$
    g_{\mu}(\mu) = \log(\mu), \qquad g_{\pi}(\pi) = \log\!\left(\frac{\pi}{1-\pi}\right).
    $$

    The probability mass function is a mixture of a point mass at zero and a Poisson
    distribution:

    $$
    P(Y = 0) = \pi + (1-\pi) e^{-\mu}, \qquad
    P(Y = k) = (1-\pi) \frac{\mu^{k} e^{-\mu}}{k!} \quad \text{for } k > 0.
    $$

    Examples
    --------
    Fit a GAMLSS for count data with excess zeros:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.4 * np.sin(x))
    pi = expit(-1.0 + 0.8 * np.cos(x))

    is_structural_zero = rng.uniform(size=n) < pi
    counts = rng.poisson(mu)
    y = np.where(is_structural_zero, 0, counts).astype(float)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "pi": "y ~ s(x)"},
        family=wk.ZeroInflatedPoisson(),
    )
    model.fit(data)
    print(model.summary())
    ```
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the distributional parameters modeled by this family.

        Returns
        -------
        tuple of str
            `("mu", "pi")` — the Poisson mean and the zero-inflation probability, both
            modeled as smooth functions of covariates.
        """
        return ("mu", "pi")

    def link(self, param: str, values: NDArray) -> NDArray:
        r"""Map a distributional parameter from its natural scale to the link scale.

        Applies the parameter-specific link function used internally by `GAMLSS`: the log
        link $\eta = \log(\mu)$ for `mu` (clamped away from zero via `_MU_FLOOR`), and the
        logit link $\eta = \log(\pi / (1-\pi))$ for `pi` (clamped away from 0 and 1).

        Parameters
        ----------
        param : str
            Name of the parameter to transform, either `"mu"` or `"pi"`.
        values : NDArray
            Values of `param` on its natural scale.

        Returns
        -------
        NDArray
            Transformed values on the link (linear predictor) scale.
        """
        if param == "mu":
            return np.log(np.maximum(values, _MU_FLOOR))
        return logit(np.clip(values, _EPS, 1.0 - _EPS))

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        r"""Map a distributional parameter from the link scale back to its natural scale.

        Implements the inverse of `link`: $\mu = e^{\eta}$ (clipped to $[-30, 30]$ before
        exponentiating to avoid overflow) for `mu`, and $\pi = \mathrm{expit}(\eta)$ for `pi`.

        Parameters
        ----------
        param : str
            Name of the parameter to transform, either `"mu"` or `"pi"`.
        eta : NDArray
            Linear predictor values on the link scale.

        Returns
        -------
        NDArray
            Values of `param` on its natural scale.
        """
        if param == "mu":
            return np.exp(np.clip(eta, -30.0, 30.0))
        return expit(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        r"""Compute the derivative of the link function with respect to the natural parameter.

        Used by the IRLS fitting routine in `GAMLSS` to convert working responses between
        the link and natural scales. Returns $d\eta/d\mu = 1/\mu$ for `mu` and
        $d\eta/d\pi = 1/(\pi(1-\pi))$ for `pi`, both evaluated with the same clamping used
        in `link`.

        Parameters
        ----------
        param : str
            Name of the parameter, either `"mu"` or `"pi"`.
        values : NDArray
            Values of `param` on its natural scale.

        Returns
        -------
        NDArray
            Derivative of the link function evaluated at `values`.
        """
        if param == "mu":
            return 1.0 / np.maximum(values, _MU_FLOOR)
        v = np.clip(values, _EPS, 1.0 - _EPS)
        return 1.0 / (v * (1.0 - v))

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Compute the first derivative of the log-likelihood with respect to a parameter.

        Implements the family-specific score function used by `GAMLSS` to build the
        working response at each IRLS iteration. For zero observations the derivative
        accounts for the mixture weight of the point mass versus the Poisson component at
        zero; for positive observations it reduces to the ordinary Poisson score. For `mu`
        this is $\partial \ell / \partial \mu$, and for `pi` it is
        $\partial \ell / \partial \pi$, both derived from the zero-inflated Poisson
        log-likelihood.

        Parameters
        ----------
        param : str
            Name of the parameter to differentiate with respect to, either `"mu"` or `"pi"`.
        y : NDArray
            Observed response values.
        params : dict of str to NDArray
            Current fitted values of all distributional parameters (`"mu"` and `"pi"`).

        Returns
        -------
        NDArray
            Elementwise first derivative of the log-likelihood with respect to `param`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        exp_neg_mu = np.exp(-np.minimum(mu, 700.0))
        p0 = np.maximum(pi + (1.0 - pi) * exp_neg_mu, _EPS)

        if param == "mu":
            return np.where(
                y == 0,
                -(1.0 - pi) * exp_neg_mu / p0,
                y / mu - 1.0,
            )
        return np.where(
            y == 0,
            (1.0 - exp_neg_mu) / p0,
            -1.0 / (1.0 - pi),
        )

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Compute the (negative) second derivative of the log-likelihood.

        Implements the family-specific working weight used by `GAMLSS`'s Fisher-scoring
        updates. The returned values approximate $-\partial^2 \ell / \partial \theta^2$ for
        `theta` in `{"mu", "pi"}`, computed separately for zero and positive observations to
        reflect the zero-inflated mixture, and floored at `_EPS` to keep weights strictly
        positive.

        Parameters
        ----------
        param : str
            Name of the parameter, either `"mu"` or `"pi"`.
        y : NDArray
            Observed response values.
        params : dict of str to NDArray
            Current fitted values of all distributional parameters (`"mu"` and `"pi"`).

        Returns
        -------
        NDArray
            Elementwise working weight for `param`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        exp_neg_mu = np.exp(-np.minimum(mu, 700.0))
        p0 = np.maximum(pi + (1.0 - pi) * exp_neg_mu, _EPS)

        if param == "mu":
            w_zero = (1.0 - pi) * exp_neg_mu * (1.0 - (1.0 - pi) * exp_neg_mu / p0) / p0
            w_pos = 1.0 / mu
            return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)
        w_zero = ((1.0 - exp_neg_mu) / p0) ** 2
        w_pos = 1.0 / ((1.0 - pi) ** 2)
        return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        r"""Compute the total log-likelihood under the zero-inflated Poisson model.

        Evaluates

        $$
        \ell = \sum_{i: y_i = 0} \log\!\big(\pi_i + (1-\pi_i) e^{-\mu_i}\big)
        + \sum_{i: y_i > 0} \Big[\log(1-\pi_i) + y_i \log(\mu_i) - \mu_i - \log(y_i!)\Big],
        $$

        summing the point-mass contribution for structural zeros with the Poisson
        contribution for observed counts.

        Parameters
        ----------
        y : NDArray
            Observed response values.
        params : dict of str to NDArray
            Fitted values of `"mu"` and `"pi"` for each observation.

        Returns
        -------
        float
            Total log-likelihood summed over all observations.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        exp_neg_mu = np.exp(-np.minimum(mu, 700.0))
        ll_zero = np.log(np.maximum(pi + (1.0 - pi) * exp_neg_mu, _EPS))
        ll_pos = np.log(1.0 - pi) + y * np.log(mu) - mu - gammaln(y + 1.0)
        ll_i = np.where(y == 0, ll_zero, ll_pos)
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Generate starting values for `mu` and `pi` before the first IRLS iteration.

        Sets `mu` to the mean of the strictly positive observations (or 1.0 if none are
        positive) for every observation, and sets `pi` to half the observed fraction of
        zeros, clipped to `[0.01, 0.5]`, giving `GAMLSS` a reasonable starting point
        without any structural information about the covariates.

        Parameters
        ----------
        y : NDArray
            Observed response values.

        Returns
        -------
        dict of str to NDArray
            Initial values for `"mu"` and `"pi"`, each broadcast to the shape of `y`.
        """
        zero_frac = float(np.mean(y == 0))
        y_pos = y[y > 0]
        mu_est = float(np.mean(y_pos)) if len(y_pos) > 0 else 1.0
        return {
            "mu": np.full_like(y, mu_est, dtype=float),
            "pi": np.full_like(y, np.clip(zero_frac * 0.5, 0.01, 0.5), dtype=float),
        }

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        """Draw a random sample of counts from the fitted zero-inflated Poisson model.

        For each observation, flips a `pi`-weighted coin to decide whether it is a
        structural zero; observations that are not structural zeros are drawn from
        `Poisson(mu)`, so a sampled value can still be zero even when it was not selected
        as a structural zero.

        Parameters
        ----------
        params : dict of str to NDArray
            Fitted values of `"mu"` and `"pi"` for each observation to simulate.
        rng : object
            Random number generator exposing `uniform` and `poisson` methods, such as a
            `numpy.random.Generator`.

        Returns
        -------
        NDArray
            Simulated response values, one per row of `params`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        is_structural_zero = rng.uniform(size=len(mu)) < pi
        counts = rng.poisson(mu)
        return np.where(is_structural_zero, 0, counts).astype(float)

    def __repr__(self) -> str:
        return "ZeroInflatedPoisson(mu=log, pi=logit)"


class ZeroInflatedNegativeBinomial(GAMLSSFamily):
    r"""Zero-inflated negative binomial (ZINB) family for GAMLSS.

    `ZeroInflatedNegativeBinomial` combines the two departures from plain `Poisson` counts that
    are most common in practice: overdispersion (variance exceeding the mean, as in
    `NegativeBinomial`) and structural excess zeros (as in `ZeroInflatedPoisson`). It is
    appropriate for count data where, even after allowing for a mixture of structural and
    sampling zeros, the remaining positive counts are still more variable than a Poisson model
    would predict — for example, healthcare utilization counts, insurance claim frequencies, or
    ecological abundance data with many true absences plus overdispersed non-zero counts. `mu`
    (the NB mean) and `pi` (the zero-inflation probability) are modeled as smooth functions of
    covariates through `GAMLSS`, while the overdispersion parameter `theta` is fixed at
    construction rather than estimated per observation.

    Parameters
    ----------
    theta : float, default=1.0
        Negative binomial size (overdispersion) parameter, must be positive. Larger values mean
        less overdispersion (`theta -> infinity` recovers `ZeroInflatedPoisson`); smaller values
        mean more overdispersion among the non-structural-zero counts. Unlike `mu` and `pi`,
        `theta` is a single fixed value shared across all observations rather than modeled by a
        smooth predictor.

    Notes
    -----
    Two distributional parameters are modeled, each with its own link:

    $$
    g_{\mu}(\mu) = \log(\mu), \qquad g_{\pi}(\pi) = \log\!\left(\frac{\pi}{1-\pi}\right).
    $$

    The probability mass function is a mixture of a point mass at zero and a Negative Binomial
    distribution using the mean-size parameterization (`Var(NB) = mu + mu^2/theta`):

    $$
    P(Y = 0) = \pi + (1-\pi)\, \mathrm{NB}(0 \mid \mu, \theta), \qquad
    P(Y = k) = (1-\pi)\, \mathrm{NB}(k \mid \mu, \theta) \quad \text{for } k > 0,
    $$

    where

    $$
    \mathrm{NB}(k \mid \mu, \theta) = \binom{k+\theta-1}{k}
    \left(\frac{\theta}{\theta+\mu}\right)^{\theta} \left(\frac{\mu}{\theta+\mu}\right)^{k}.
    $$

    Examples
    --------
    Fit a GAMLSS for overdispersed count data with excess zeros:

    ```{python}
    import numpy as np
    import whittaker as wk
    from scipy.special import expit

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(0.5 + 0.4 * np.sin(x))
    pi = expit(-1.0 + 0.8 * np.cos(x))
    theta = 2.0

    is_structural_zero = rng.uniform(size=n) < pi
    counts = rng.negative_binomial(theta, theta / (theta + mu))
    y = np.where(is_structural_zero, 0, counts).astype(float)

    data = {"x": x, "y": y}

    model = wk.GAMLSS(
        formulas={"mu": "y ~ s(x)", "pi": "y ~ s(x)"},
        family=wk.ZeroInflatedNegativeBinomial(theta=theta),
    )
    model.fit(data)
    print(model.summary())
    ```
    """

    def __init__(self, theta: float = 1.0) -> None:
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}.")
        self._theta = float(theta)

    @property
    def theta(self) -> float:
        """Fixed negative binomial size (overdispersion) parameter.

        Returns
        -------
        float
            The value supplied at construction; smaller values indicate more
            overdispersion relative to the Poisson case, which is recovered as
            `theta -> infinity`.
        """
        return self._theta

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the distributional parameters modeled by this family.

        Returns
        -------
        tuple of str
            `("mu", "pi")` — the negative binomial mean and the zero-inflation
            probability, both modeled as smooth functions of covariates. The
            overdispersion parameter `theta` is fixed at construction and is not part of
            this tuple.
        """
        return ("mu", "pi")

    def link(self, param: str, values: NDArray) -> NDArray:
        r"""Map a distributional parameter from its natural scale to the link scale.

        Applies the log link $\eta = \log(\mu)$ for `mu` (clamped away from zero via
        `_MU_FLOOR`), and the logit link $\eta = \log(\pi / (1-\pi))$ for `pi` (clamped
        away from 0 and 1). The overdispersion parameter `theta` is fixed and has no link.

        Parameters
        ----------
        param : str
            Name of the parameter to transform, either `"mu"` or `"pi"`.
        values : NDArray
            Values of `param` on its natural scale.

        Returns
        -------
        NDArray
            Transformed values on the link (linear predictor) scale.
        """
        if param == "mu":
            return np.log(np.maximum(values, _MU_FLOOR))
        return logit(np.clip(values, _EPS, 1.0 - _EPS))

    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        r"""Map a distributional parameter from the link scale back to its natural scale.

        Implements the inverse of `link`: $\mu = e^{\eta}$ (clipped to $[-30, 30]$ before
        exponentiating to avoid overflow) for `mu`, and $\pi = \mathrm{expit}(\eta)$ for `pi`.

        Parameters
        ----------
        param : str
            Name of the parameter to transform, either `"mu"` or `"pi"`.
        eta : NDArray
            Linear predictor values on the link scale.

        Returns
        -------
        NDArray
            Values of `param` on its natural scale.
        """
        if param == "mu":
            return np.exp(np.clip(eta, -30.0, 30.0))
        return expit(eta)

    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        r"""Compute the derivative of the link function with respect to the natural parameter.

        Returns $d\eta/d\mu = 1/\mu$ for `mu` and $d\eta/d\pi = 1/(\pi(1-\pi))$ for `pi`,
        using the same clamping applied in `link`, for use by `GAMLSS`'s IRLS fitting
        routine.

        Parameters
        ----------
        param : str
            Name of the parameter, either `"mu"` or `"pi"`.
        values : NDArray
            Values of `param` on its natural scale.

        Returns
        -------
        NDArray
            Derivative of the link function evaluated at `values`.
        """
        if param == "mu":
            return 1.0 / np.maximum(values, _MU_FLOOR)
        v = np.clip(values, _EPS, 1.0 - _EPS)
        return 1.0 / (v * (1.0 - v))

    def _nb_log_pmf(self, k: NDArray, mu: NDArray) -> NDArray:
        theta = self._theta
        return (
            gammaln(k + theta)
            - gammaln(k + 1.0)
            - gammaln(theta)
            + theta * np.log(theta / (mu + theta))
            + k * np.log(mu / (mu + theta))
        )

    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Compute the first derivative of the log-likelihood with respect to a parameter.

        Implements the family-specific score function for the zero-inflated negative
        binomial model, combining the derivative of the `NB(0 | mu, theta)` probability
        weighted by the mixture for zero observations with the ordinary negative binomial
        score $\partial \ell / \partial \mu = (y - \mu) / (\mu (1 + \mu/\theta))$ for
        positive observations. For `pi` the score reflects the same zero/positive split as
        in `ZeroInflatedPoisson.dl_dtheta`.

        Parameters
        ----------
        param : str
            Name of the parameter to differentiate with respect to, either `"mu"` or `"pi"`.
        y : NDArray
            Observed response values.
        params : dict of str to NDArray
            Current fitted values of `"mu"` and `"pi"`.

        Returns
        -------
        NDArray
            Elementwise first derivative of the log-likelihood with respect to `param`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        theta = self._theta

        nb_p0 = np.exp(self._nb_log_pmf(np.zeros_like(y), mu))
        p0 = np.maximum(pi + (1.0 - pi) * nb_p0, _EPS)

        if param == "mu":
            dnb_dmu_at_0 = nb_p0 * (-theta / (mu + theta))
            dl_zero = (1.0 - pi) * dnb_dmu_at_0 / p0
            dl_pos = (y - mu) / (mu * (1.0 + mu / theta))
            return np.where(y == 0, dl_zero, dl_pos)

        dl_zero = (1.0 - nb_p0) / p0
        dl_pos = -1.0 / (1.0 - pi)
        return np.where(y == 0, dl_zero, dl_pos)

    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        r"""Compute the (negative) second derivative of the log-likelihood.

        Implements the family-specific working weight used by `GAMLSS`'s Fisher-scoring
        updates for the zero-inflated negative binomial model, computed separately for
        zero and positive observations and floored at `_EPS` to keep weights strictly
        positive.

        Parameters
        ----------
        param : str
            Name of the parameter, either `"mu"` or `"pi"`.
        y : NDArray
            Observed response values.
        params : dict of str to NDArray
            Current fitted values of `"mu"` and `"pi"`.

        Returns
        -------
        NDArray
            Elementwise working weight for `param`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        theta = self._theta

        nb_p0 = np.exp(self._nb_log_pmf(np.zeros_like(y), mu))
        p0 = np.maximum(pi + (1.0 - pi) * nb_p0, _EPS)

        if param == "mu":
            w_pos = 1.0 / (mu * (1.0 + mu / theta))
            dnb = nb_p0 * theta / (mu + theta)
            w_zero = (1.0 - pi) * dnb * (1.0 - (1.0 - pi) * dnb / p0) / p0
            return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)

        w_zero = ((1.0 - nb_p0) / p0) ** 2
        w_pos = 1.0 / ((1.0 - pi) ** 2)
        return np.maximum(np.where(y == 0, w_zero, w_pos), _EPS)

    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        r"""Compute the total log-likelihood under the zero-inflated negative binomial model.

        Evaluates

        $$
        \ell = \sum_{i: y_i = 0} \log\!\big(\pi_i + (1-\pi_i)\, \mathrm{NB}(0 \mid \mu_i, \theta)\big)
        + \sum_{i: y_i > 0} \Big[\log(1-\pi_i) + \log \mathrm{NB}(y_i \mid \mu_i, \theta)\Big],
        $$

        where $\mathrm{NB}(\cdot \mid \mu, \theta)$ is the negative binomial probability
        mass function in the mean-size parameterization with the fixed `theta`.

        Parameters
        ----------
        y : NDArray
            Observed response values.
        params : dict of str to NDArray
            Fitted values of `"mu"` and `"pi"` for each observation.

        Returns
        -------
        float
            Total log-likelihood summed over all observations.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)

        nb_log_p0 = self._nb_log_pmf(np.zeros_like(y), mu)
        ll_zero = np.log(np.maximum(pi + (1.0 - pi) * np.exp(nb_log_p0), _EPS))
        ll_pos = np.log(1.0 - pi) + self._nb_log_pmf(y, mu)
        ll_i = np.where(y == 0, ll_zero, ll_pos)
        return float(np.sum(ll_i))

    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Generate starting values for `mu` and `pi` before the first IRLS iteration.

        Sets `mu` to the mean of the strictly positive observations (or 1.0 if none are
        positive) for every observation, and sets `pi` to half the observed fraction of
        zeros, clipped to `[0.01, 0.5]`. The fixed overdispersion parameter `theta` is not
        part of the returned dictionary since it is not estimated per observation.

        Parameters
        ----------
        y : NDArray
            Observed response values.

        Returns
        -------
        dict of str to NDArray
            Initial values for `"mu"` and `"pi"`, each broadcast to the shape of `y`.
        """
        zero_frac = float(np.mean(y == 0))
        y_pos = y[y > 0]
        mu_est = float(np.mean(y_pos)) if len(y_pos) > 0 else 1.0
        return {
            "mu": np.full_like(y, mu_est, dtype=float),
            "pi": np.full_like(y, np.clip(zero_frac * 0.5, 0.01, 0.5), dtype=float),
        }

    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        """Draw a random sample of counts from the fitted zero-inflated NB model.

        For each observation, flips a `pi`-weighted coin to decide whether it is a
        structural zero; observations that are not structural zeros are drawn from a
        negative binomial distribution with the fixed `theta` and success probability
        `theta / (mu + theta)`, so a sampled value can still be zero even when it was not
        selected as a structural zero.

        Parameters
        ----------
        params : dict of str to NDArray
            Fitted values of `"mu"` and `"pi"` for each observation to simulate.
        rng : object
            Random number generator exposing `uniform` and `negative_binomial` methods,
            such as a `numpy.random.Generator`.

        Returns
        -------
        NDArray
            Simulated response values, one per row of `params`.
        """
        mu = np.maximum(params["mu"], _MU_FLOOR)
        pi = np.clip(params["pi"], _EPS, 1.0 - _EPS)
        theta = self._theta
        p = theta / (mu + theta)
        is_structural_zero = rng.uniform(size=len(mu)) < pi
        counts = rng.negative_binomial(theta, p)
        return np.where(is_structural_zero, 0, counts).astype(float)

    def __repr__(self) -> str:
        return f"ZeroInflatedNegativeBinomial(theta={self._theta:.4g}, mu=log, pi=logit)"
