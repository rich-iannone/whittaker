"""Cox proportional hazards family for survival analysis.

Implements a partial-likelihood approach where the hazard is modeled as:

    h(t | x) = h0(t) * exp(eta)

The response `y` encodes survival times. The event/censoring indicator is passed separately via the
`status` parameter (a column name resolved from the data dict).

Supports `Breslow` and `Efron` tie-handling approximations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.families.base import Family

_EPS = np.finfo(float).eps
_ETA_CLIP = 20.0


class CoxPH(Family):
    r"""Cox proportional hazards family for survival analysis.

    `CoxPH` fits a semiparametric proportional hazards model, allowing smooth (via `s()`) and
    linear covariate effects on the log hazard while leaving the baseline hazard `h0(t)`
    unspecified. Use this family whenever the response is a time-to-event outcome that may be
    right-censored — for example, time to failure, time to churn, or time to death/relapse in
    survival data — and the goal is to model how covariates shift the instantaneous risk of the
    event over time. Rather than a per-observation deviance and log-likelihood in the usual GLM
    sense, `CoxPH` maximizes the Cox partial likelihood via a custom `irls_update`, so the
    "response" `y` passed to `GAM.fit()` is the observed survival/censoring time, and the event
    indicator is supplied separately through `set_data()` (populated automatically by `GAM.fit()`
    from the column named by `status`).

    Parameters
    ----------
    status : str, default="event"
        Name of the column in the data dict containing the event indicator (`1` = event
        observed, `0` = right-censored). This column is looked up automatically from the data
        passed to `GAM.fit()`.
    ties : str, default="breslow"
        Tie-handling method for the partial likelihood when multiple observations share the same
        event time: `"breslow"` (default, simpler and faster) or `"efron"` (more accurate when
        ties are frequent).

    Notes
    -----
    The hazard is modeled multiplicatively as

    $$
    h(t \mid x) = h_0(t)\, e^{\eta(x)}, \qquad \eta(x) = X\beta,
    $$

    where `h0(t)` is an unspecified baseline hazard and `eta` is the (possibly smooth) linear
    predictor, so `link` and `link_inverse` are both the identity. There is no closed-form
    variance function or unit deviance in the usual GLM sense; instead, model fitting maximizes
    the Cox partial log-likelihood,

    $$
    \ell(\beta) = \sum_{i:\, \delta_i = 1} \left[ \eta_i - \log\!\left( \sum_{j \in R(t_i)} e^{\eta_j} \right) \right],
    $$

    where $\delta_i$ is the event indicator and $R(t_i)$ is the risk set at time $t_i$ (those
    still under observation just before $t_i$). The overall "deviance" reported by
    `GAM.summary()` is $-2\ell(\beta)$. After fitting, `baseline_hazard()` and
    `survival_function()` expose the Breslow estimate of the cumulative baseline hazard and the
    implied survival curve.

    Examples
    --------
    Fit a Cox proportional hazards GAM with a smooth effect of age on the hazard:

    ```{python}
    import numpy as np
    import whittaker as wk

    rng = np.random.default_rng(0)
    n = 300
    age = rng.uniform(40, 80, n)
    risk = np.exp(0.03 * (age - 60))
    time = rng.exponential(1.0 / risk)
    censor_time = rng.exponential(2.0, n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(float)

    data = {"time": observed_time, "age": age, "event": event}

    model = wk.GAM("time ~ s(age)", family=wk.CoxPH(status="event"))
    model.fit(data, method="REML")
    print(model.summary())
    ```
    """

    def __init__(self, status: str = "event", ties: str = "breslow") -> None:
        if ties not in ("breslow", "efron"):
            raise ValueError(f"ties must be 'breslow' or 'efron', got {ties!r}.")
        self._status_col = status
        self._ties = ties
        self._event: NDArray | None = None
        self._time: NDArray | None = None
        self._sort_idx: NDArray | None = None
        self._baseline_cumhaz: NDArray | None = None
        self._baseline_times: NDArray | None = None

    @property
    def ties(self) -> str:
        return self._ties

    def set_data(self, data: dict[str, NDArray]) -> None:
        if self._status_col not in data:
            available = ", ".join(sorted(data))
            raise KeyError(
                f"Status column {self._status_col!r} not found in data. "
                f"Available columns: {available}."
            )
        self._event = np.asarray(data[self._status_col], dtype=float)

    def link(self, mu: NDArray) -> NDArray:
        return mu

    def link_inverse(self, eta: NDArray) -> NDArray:
        return eta

    def link_derivative(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def variance(self, mu: NDArray) -> NDArray:
        return np.ones_like(mu)

    def initialize(self, y: NDArray) -> NDArray:
        self._time = np.asarray(y, dtype=float)
        self._sort_idx = np.argsort(self._time)
        return np.zeros_like(y)

    def _partial_log_likelihood(self, eta: NDArray) -> float:
        if self._event is None or self._sort_idx is None:
            return 0.0

        order = self._sort_idx
        eta_s = eta[order]
        event_s = self._event[order]
        exp_eta_s = np.exp(np.clip(eta_s, -_ETA_CLIP, _ETA_CLIP))

        if self._ties == "breslow":
            S0 = np.cumsum(exp_eta_s[::-1])[::-1]
            return float(np.sum(event_s * (eta_s - np.log(np.maximum(S0, _EPS)))))

        return self._efron_pll(eta_s, event_s, exp_eta_s)

    def _efron_pll(self, eta_s: NDArray, event_s: NDArray, exp_eta_s: NDArray) -> float:
        time_s = self._time[self._sort_idx]
        n = len(eta_s)
        S0_full = np.cumsum(exp_eta_s[::-1])[::-1]

        pll = 0.0
        i = 0
        while i < n:
            if event_s[i] == 0:
                i += 1
                continue

            t_cur = time_s[i]
            tied_start = i
            event_indices = []
            while i < n and time_s[i] == t_cur:
                if event_s[i] == 1:
                    event_indices.append(i)
                i += 1

            d_k = len(event_indices)
            s_k = sum(exp_eta_s[j] for j in event_indices)

            for j in event_indices:
                pll += eta_s[j]

            for r in range(d_k):
                denom = S0_full[tied_start] - (r / d_k) * s_k
                pll -= np.log(max(denom, _EPS))

        return pll

    def irls_update(self, y: NDArray, mu: NDArray, eta: NDArray) -> tuple[NDArray, NDArray]:
        if self._event is None:
            raise RuntimeError("CoxPH requires set_data() before fitting.")

        order = self._sort_idx
        eta_s = eta[order]
        event_s = self._event[order]
        exp_eta_s = np.exp(np.clip(eta_s, -_ETA_CLIP, _ETA_CLIP))

        if self._ties == "breslow":
            gradient_s, hess_diag_s = self._breslow_grad_hess(event_s, exp_eta_s)
        else:
            gradient_s, hess_diag_s = self._efron_grad_hess(event_s, exp_eta_s)

        inv_order = np.argsort(order)
        gradient = gradient_s[inv_order]
        W = np.maximum(hess_diag_s[inv_order], _EPS)
        z = eta + gradient / W

        self._compute_baseline_hazard(eta)

        return z, W

    def _breslow_grad_hess(self, event_s: NDArray, exp_eta_s: NDArray) -> tuple[NDArray, NDArray]:
        S0 = np.cumsum(exp_eta_s[::-1])[::-1]

        inv_S0_event = event_s / np.maximum(S0, _EPS)
        cum_inv_S0 = np.cumsum(inv_S0_event)

        inv_S0_sq_event = event_s / np.maximum(S0**2, _EPS)
        cum_inv_S0_sq = np.cumsum(inv_S0_sq_event)

        gradient_s = event_s - exp_eta_s * cum_inv_S0
        hess_diag_s = exp_eta_s * cum_inv_S0 - exp_eta_s**2 * cum_inv_S0_sq

        return gradient_s, hess_diag_s

    def _efron_grad_hess(self, event_s: NDArray, exp_eta_s: NDArray) -> tuple[NDArray, NDArray]:
        time_s = self._time[self._sort_idx]
        n = len(event_s)
        S0_full = np.cumsum(exp_eta_s[::-1])[::-1]

        inc_a = np.zeros(n)
        inc_b = np.zeros(n)

        event_time_data: list[tuple] = []

        i = 0
        while i < n:
            if event_s[i] == 0:
                i += 1
                continue

            t_cur = time_s[i]
            tied_start = i
            event_indices = []
            while i < n and time_s[i] == t_cur:
                if event_s[i] == 1:
                    event_indices.append(i)
                i += 1

            d_k = len(event_indices)
            s_k = sum(exp_eta_s[j] for j in event_indices)

            a_k = 0.0
            b_k = 0.0
            c_k = 0.0
            e_k = 0.0

            for r in range(d_k):
                denom = max(S0_full[tied_start] - (r / d_k) * s_k, _EPS)
                a_k += 1.0 / denom
                b_k += 1.0 / denom**2
                c_k += (r / d_k) / denom
                e_k += (1.0 - r / d_k) ** 2 / denom**2

            inc_a[tied_start] += a_k
            inc_b[tied_start] += b_k
            event_time_data.append((d_k, a_k, b_k, c_k, e_k, event_indices))

        cum_a = np.cumsum(inc_a)
        cum_b = np.cumsum(inc_b)

        gradient_s = event_s.astype(float) - exp_eta_s * cum_a
        hess_diag_s = exp_eta_s * cum_a - exp_eta_s**2 * cum_b

        for d_k, a_k, b_k, c_k, e_k, event_indices in event_time_data:
            if d_k <= 1:
                continue
            for j in event_indices:
                gradient_s[j] += exp_eta_s[j] * c_k
                hess_diag_s[j] += -exp_eta_s[j] * c_k + exp_eta_s[j] ** 2 * (b_k - e_k)

        return gradient_s, hess_diag_s

    def deviance(self, y: NDArray, mu: NDArray, *, weights: NDArray | None = None) -> float:
        eta = mu
        pll = self._partial_log_likelihood(eta)
        return -2.0 * pll

    def unit_deviance(self, y: NDArray, mu: NDArray) -> NDArray:
        return np.ones_like(y)

    def log_likelihood(
        self, y: NDArray, mu: NDArray, scale: float, *, weights: NDArray | None = None
    ) -> float:
        eta = mu
        return self._partial_log_likelihood(eta)

    @property
    def scale_known(self) -> bool:
        return True

    def simulate(self, mu: NDArray, scale: float, rng: object) -> NDArray:
        if self._baseline_cumhaz is None:
            raise RuntimeError("Model must be fitted before simulation.")
        eta = mu
        exp_eta = np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))
        u = rng.uniform(0, 1, len(eta))
        target_cumhaz = -np.log(np.maximum(u, _EPS)) / np.maximum(exp_eta, _EPS)

        times = np.full(len(eta), self._baseline_times[-1])
        for i in range(len(eta)):
            idx = np.searchsorted(self._baseline_cumhaz, target_cumhaz[i])
            if idx < len(self._baseline_times):
                times[i] = self._baseline_times[idx]
        return times

    def _compute_baseline_hazard(self, eta: NDArray) -> None:
        if self._event is None or self._sort_idx is None:
            return

        order = self._sort_idx
        time_s = self._time[order]
        event_s = self._event[order]
        exp_eta_s = np.exp(np.clip(eta[order], -_ETA_CLIP, _ETA_CLIP))

        S0 = np.cumsum(exp_eta_s[::-1])[::-1]

        unique_event_times = []
        cumhaz_values = []
        cumhaz = 0.0

        i = 0
        n = len(time_s)
        while i < n:
            if event_s[i] == 0:
                i += 1
                continue

            t_cur = time_s[i]
            d_k = 0
            while i + d_k < n and time_s[i + d_k] == t_cur and event_s[i + d_k] == 1:
                d_k += 1

            cumhaz += d_k / max(S0[i], _EPS)
            unique_event_times.append(t_cur)
            cumhaz_values.append(cumhaz)
            i += d_k

        if unique_event_times:
            self._baseline_times = np.array(unique_event_times)
            self._baseline_cumhaz = np.array(cumhaz_values)

    def baseline_hazard(self) -> tuple[NDArray, NDArray]:
        if self._baseline_times is None or self._baseline_cumhaz is None:
            raise RuntimeError("Baseline hazard not computed. Fit the model first.")
        return self._baseline_times.copy(), self._baseline_cumhaz.copy()

    def survival_function(self, eta: NDArray) -> NDArray:
        if self._baseline_cumhaz is None:
            raise RuntimeError("Baseline hazard not computed. Fit the model first.")
        exp_eta = np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))
        H0 = self._baseline_cumhaz[-1]
        return np.exp(-H0 * exp_eta)

    def __repr__(self) -> str:
        return f"CoxPH(ties={self._ties!r})"
