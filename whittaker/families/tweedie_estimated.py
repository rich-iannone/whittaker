r"""Tweedie family with estimated variance power parameter.

Provides `tw()` which creates a Tweedie family whose variance power `p` is estimated from the data
by profile likelihood. During `GAM.fit()`, the model is fitted at a grid of candidate `p` values and
the one minimising AIC (or another criterion) is selected.
"""

from __future__ import annotations

from whittaker.families.tweedie import Tweedie


class TweedieEstimated(Tweedie):
    r"""Tweedie family with variance power estimated by profile likelihood.

    `TweedieEstimated` behaves exactly like `Tweedie` (log link, `V(mu) = mu^p`, compound
    Poisson-Gamma density for `1 < p < 2`) except that the variance power `p` is not fixed at
    construction time. Instead, `GAM.fit()` performs a profile-likelihood grid search: the model
    is refit at each candidate `p` in `p_range`, and the value minimizing AIC is retained. This
    is useful when the appropriate degree of "Poisson-ness" versus "Gamma-ness" in claims,
    rainfall, or other zero-inflated positive data is not known in advance. Users typically
    construct this family via the `tw()` convenience function rather than instantiating
    `TweedieEstimated` directly.

    Parameters
    ----------
    p_range : tuple of float, default=(1.01, 1.99)
        `(p_min, p_max)` range to search over. Both endpoints should lie strictly within `(1, 2)`
        for the compound Poisson-Gamma case (the typical use case for insurance-style data with
        structural zeros).
    n_grid : int, default=20
        Number of candidate `p` values in the initial grid search over `p_range`. A finer grid
        gives a more precise estimate of `p` at the cost of additional model fits.

    Notes
    -----
    Once fitted, the estimated value of `p` is available via the inherited `p` property, and
    `p_estimated` reports whether estimation has completed. The link, variance function, and
    deviance are identical to those of `Tweedie`:

    $$
    g(\mu) = \log(\mu), \qquad V(\mu) = \mu^{p}.
    $$

    See `Tweedie` for the full deviance and log-likelihood formulas.

    Examples
    --------
    Fit a GAM letting Whittaker choose the Tweedie variance power automatically:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    x = np.linspace(0, 2 * np.pi, n)
    mu = np.exp(1.0 + 0.5 * np.sin(x))
    y = np.array([rng.gamma(2.0, m / 2.0) if rng.random() > 0.3 else 0.0 for m in mu])

    data = {"x": x, "y": y}

    model = wk.GAM("y ~ s(x)", family=wk.tw())
    model.fit(data, method="REML")
    print(model.family)
    print(model.summary())
    ```
    """

    def __init__(
        self,
        p_range: tuple[float, float] = (1.01, 1.99),
        n_grid: int = 20,
    ) -> None:
        self._p_range = p_range
        self._n_grid = n_grid
        self._p_estimated = False
        super().__init__(p=sum(p_range) / 2.0)

    @property
    def p_estimated(self) -> bool:
        """Whether the variance power `p` has completed profile-likelihood estimation.

        `False` immediately after construction, when `p` is only a provisional midpoint of
        `p_range`. Set to `True` by `_set_p` once `GAM.fit()` has run its grid search over
        `p_range` and selected the AIC-minimizing value, at which point the inherited `p`
        property reports the estimated value rather than the placeholder.
        """
        return self._p_estimated

    def _set_p(self, p: float) -> None:
        self._p = float(p)
        self._p_estimated = True

    def __repr__(self) -> str:
        if self._p_estimated:
            return f"Tweedie(p={self._p:.4f}, link='log', estimated=True)"
        return f"Tweedie(p=?, link='log', range={self._p_range})"


def tw(
    p_range: tuple[float, float] = (1.01, 1.99),
    n_grid: int = 20,
) -> TweedieEstimated:
    r"""Create a Tweedie family with estimated variance power.

    Convenience constructor mirroring the `tw()` function familiar from `mgcv`. The variance
    power `p` of the Tweedie distribution (see `Tweedie`) is selected automatically by profile
    likelihood during model fitting rather than fixed by the user. The model is fitted at
    `n_grid` candidate values of `p` spaced across `p_range`, and the value minimizing AIC is
    chosen as the final family.

    Parameters
    ----------
    p_range : tuple of float, default=(1.01, 1.99)
        `(p_min, p_max)` range to search. Must satisfy `1 < p_min` and `p_max < 2` (or both `> 2`
        for the positive-continuous case). Defaults to `(1.01, 1.99)`, which covers the compound
        Poisson-Gamma case used for most zero-inflated positive data.
    n_grid : int, default=20
        Number of candidate `p` values in the grid search. Defaults to `20`.

    Returns
    -------
    TweedieEstimated
        A Tweedie family, with variance function $V(\mu) = \mu^{p}$ and log link
        $g(\mu) = \log(\mu)$, whose power `p` will be estimated by profile likelihood the next
        time the returned family is passed to `GAM.fit()`.

    Examples
    --------
    ```{python}
    import whittaker as wk

    model = wk.GAM("y ~ s(x)", family=wk.tw(p_range=(1.05, 1.95), n_grid=15))
    ```
    """
    return TweedieEstimated(p_range=p_range, n_grid=n_grid)
