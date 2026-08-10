"""Tweedie family with estimated variance power parameter.

Provides `tw()` which creates a Tweedie family whose variance power `p` is estimated from the data
by profile likelihood. During `GAM.fit()`, the model is fitted at a grid of candidate `p` values and
the one minimising AIC (or another criterion) is selected.
"""

from __future__ import annotations

from whittaker.families.tweedie import Tweedie


class TweedieEstimated(Tweedie):
    """Tweedie family with variance power estimated by profile likelihood.

    Parameters
    ----------
    p_range:
        `(p_min, p_max)` range to search over. Defaults to `(1.01, 1.99)`.
    n_grid:
        Number of candidate `p` values in the initial grid search.
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
    """Create a Tweedie family with estimated variance power.

    The variance power `p` is selected by profile likelihood during model fitting. The model is
    fitted at `n_grid` candidate values of `p` in `p_range`, and the value minimising AIC is chosen.

    Parameters
    ----------
    p_range:
        `(p_min, p_max)` range to search. Must satisfy `1 < p_min` and `p_max < 2` (or both > 2 for
        the positive-continuous case). Defaults to `(1.01, 1.99)` for compound Poisson-Gamma.
    n_grid:
        Number of candidate `p` values. Defaults to `20`.

    Returns
    -------
    TweedieEstimated
        A Tweedie family whose `p` will be estimated during fitting.
    """
    return TweedieEstimated(p_range=p_range, n_grid=n_grid)
