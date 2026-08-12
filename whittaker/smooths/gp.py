"""Gaussian process smooth basis.

Implements smooths based on Gaussian process covariance functions. The basis is constructed from
the leading eigenfunctions of the covariance matrix evaluated at training knot locations.

Supported covariance functions (selected via the `cov=` parameter):

* `"exp"`: exponential (Matérn ν=½): `σ² exp(-r/ρ)`
* `"matern32"`: Matérn ν=3/2: `σ² (1 + √3 r/ρ) exp(-√3 r/ρ)`
* `"matern52"`: Matérn ν=5/2: `σ² (1 + √5 r/ρ + 5r²/(3ρ²)) exp(-√5 r/ρ)`
* `"sqexp"`: squared exponential (RBF): `σ² exp(-r²/(2ρ²))`

Usage in a formula::

    s(x, bs="gp")
    s(x1, x2, bs="gp", xt="matern32")
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from whittaker.smooths.base import SmoothBasis

_EPS = np.finfo(float).eps

_COV_FUNCTIONS = {"exp", "matern32", "matern52", "sqexp"}


def _pairwise_distances(x1: NDArray, x2: NDArray) -> NDArray:
    diff = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1) + _EPS)


def _cov_matrix(x1: NDArray, x2: NDArray, cov: str, rho: float) -> NDArray:
    r = _pairwise_distances(x1, x2)
    scaled = r / rho

    if cov == "exp":
        return np.exp(-scaled)
    elif cov == "matern32":
        s3 = np.sqrt(3.0) * scaled
        return (1.0 + s3) * np.exp(-s3)
    elif cov == "matern52":
        s5 = np.sqrt(5.0) * scaled
        return (1.0 + s5 + s5**2 / 3.0) * np.exp(-s5)
    elif cov == "sqexp":
        return np.exp(-0.5 * scaled**2)
    else:
        raise ValueError(
            f"Unknown covariance function {cov!r}. Supported: {sorted(_COV_FUNCTIONS)}"
        )


class GaussianProcess(SmoothBasis):
    r"""Gaussian process (kriging) smooth basis.

    Equivalent to mgcv's `bs="gp"` basis. This basis treats the unknown smooth function as a
    realization of a zero-mean Gaussian process with a chosen covariance (kernel) function, in the
    spirit of kriging/spatial statistics. Rather than working with the full `n x n` covariance
    matrix (which does not scale well and has no natural low-rank truncation-by-penalty like TPRS),
    this implementation builds a rank-`k` basis from the leading eigenfunctions of the covariance
    matrix evaluated at the training points, with the inverse eigenvalues serving directly as the
    penalty. Because the covariance function is stationary and isotropic (depends only on distance
    between points), `GaussianProcess` is naturally suited to spatial covariates or any setting
    where you want smoothness governed by a physically or statistically motivated correlation
    structure — e.g. exponential decay of spatial correlation — rather than a derivative-based
    bending-energy penalty like TPRS.

    Parameters
    ----------
    k:
        Number of basis functions, i.e. the number of leading eigenfunctions of the covariance
        matrix retained. Larger `k` captures more of the covariance structure at the cost of more
        computation; if `k` exceeds the number of training points `n`, it is silently reduced to
        `n`. The default is `10`.
    cov:
        Name of the covariance (kernel) function used to build the Gram matrix. One of:

        * `"exp"` — exponential covariance (Matern with `nu=1/2`): `sigma^2 exp(-r / rho)`.
          Produces rough, non-differentiable sample paths; use when the underlying process is
          expected to be continuous but not smooth.
        * `"matern32"` — Matern with `nu=3/2`: once-differentiable sample paths. A reasonable
          general-purpose default, balancing smoothness and local flexibility. This is the default.
        * `"matern52"` — Matern with `nu=5/2`: twice-differentiable sample paths, smoother than
          `"matern32"`.
        * `"sqexp"` — squared exponential (RBF): `sigma^2 exp(-r^2 / (2 rho^2))`. Produces
          infinitely differentiable, very smooth sample paths; can over-smooth sharp local features.

    Notes
    -----
    Given training covariates `x` with pairwise distances `r = ||x_i - x_j||`, the covariance
    (Gram) matrix `C` has entries `C_{ij} = k(r_{ij}; rho)` for the chosen kernel `k`. The range
    parameter `rho` is not user-specified; it is set automatically during `fit()` to one quarter of
    the mean range of the covariates, a simple heuristic that keeps the effective correlation length
    commensurate with the spread of the data. `C` is eigendecomposed and the `k` eigenvectors `U`
    with the largest eigenvalues `d_1, ..., d_k` are retained:

    $$
    \mathbf{C} \approx \mathbf{U} \operatorname{diag}(d_1, \ldots, d_k) \mathbf{U}^\top .
    $$

    The basis functions evaluated at new points `x*` are

    $$
    \mathbf{B}(x^*) = \mathbf{C}(x^*, x_{\text{train}}) \, \mathbf{U} \,
    \operatorname{diag}(d_1, \ldots, d_k)^{-1},
    $$

    i.e. the covariance between `x*` and the training points, projected onto the retained
    eigenvectors and rescaled by the inverse eigenvalues (a Nystrom-style low-rank Karhunen-Loeve
    approximation to the process). The penalty matrix is diagonal in the inverse eigenvalues,

    $$
    \mathbf{S} = \operatorname{diag}(d_1^{-1}, \ldots, d_k^{-1}),
    $$

    which corresponds to the negative log-density of the Gaussian process prior on the coefficients:
    directions with small eigenvalue (little prior variance) are penalized heavily, and directions
    with large eigenvalue are penalized lightly. Because every eigenvalue is penalized,
    `null_space_dimension()` is `0` — there is no unpenalized null space, unlike thin plate or cubic
    regression splines, so even the "constant" and "linear" trends across the domain are
    (lightly) shrunk under this basis. Eigenvalues are clamped away from zero (to machine epsilon)
    for numerical stability when inverting; using a very small `k` or a badly-scaled covariate range
    can still lead to an ill-conditioned Gram matrix.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import GaussianProcess

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)

    basis = GaussianProcess(k=10, cov="matern32").fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
    """

    def __init__(self, k: int = 10, cov: str = "matern32") -> None:
        if cov not in _COV_FUNCTIONS:
            raise ValueError(f"Unknown covariance {cov!r}. Supported: {sorted(_COV_FUNCTIONS)}")
        self._k = k
        self._cov = cov
        self._fitted = False

        self._d: int = 0
        self._x_train: NDArray
        self._rho: float = 1.0
        self._U: NDArray  # (n_train, k) eigenvectors
        self._eigenvalues: NDArray  # (k,) eigenvalues

    def fit(self, x: NDArray) -> GaussianProcess:
        """Fit the Gaussian process basis to training data `x`.

        Builds the covariance matrix at the training points, sets the range parameter `rho`
        automatically from the spread of the covariates, and eigendecomposes the covariance matrix
        to obtain the leading `k` eigenfunctions.

        Parameters
        ----------
        x:
            Training covariates. Shape `(n,)` for univariate or `(n, d)` for multivariate.

        Returns
        -------
        GaussianProcess
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If the number of observations `n` is smaller than `k`.
        """
        x2d = self._as_2d(x)
        n, d = x2d.shape

        if n < self._k:
            raise ValueError(f"n={n} < k={self._k}. Reduce k to at most {n - 1}.")

        self._d = d
        self._x_train = x2d.copy()

        ranges = x2d.max(axis=0) - x2d.min(axis=0)
        self._rho = float(np.mean(ranges[ranges > _EPS])) / 4.0
        if self._rho < _EPS:
            self._rho = 1.0

        C = _cov_matrix(x2d, x2d, self._cov, self._rho)
        C = (C + C.T) * 0.5

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = min(self._k, n)
        self._U = eigenvectors[:, :k]
        self._eigenvalues = np.maximum(eigenvalues[:k], _EPS)
        self._k = k

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the Gaussian process basis at `x`.

        Parameters
        ----------
        x:
            Covariate values. Shape `(n,)` or `(n, d)` where `d` must match the training dimension.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)`, equal to the covariance between `x` and the training
            points projected onto the retained eigenvectors and rescaled by the inverse eigenvalues.

        Raises
        ------
        ValueError
            If the covariate dimension of `x` does not match the training dimension.
        """
        self._check_fitted()
        x2d = self._as_2d(x)

        if x2d.shape[1] != self._d:
            raise ValueError(f"Expected {self._d} covariate(s), got {x2d.shape[1]}.")

        C_new = _cov_matrix(x2d, self._x_train, self._cov, self._rho)
        B = C_new @ self._U / self._eigenvalues[np.newaxis, :]
        return B

    def penalty_matrix(self) -> NDArray:
        """Return the `k x k` penalty matrix `S = diag(1 / eigenvalues)`.

        Returns
        -------
        NDArray
            Diagonal matrix of shape `(k, k)` whose entries are the inverse eigenvalues of the
            covariance matrix retained during `fit()`. There is no unpenalized block: this matrix
            is strictly positive definite.
        """
        self._check_fitted()
        return np.diag(1.0 / self._eigenvalues)

    def null_space_dimension(self) -> int:
        """Return `0`: the Gaussian process penalty has no unpenalized null space.

        Returns
        -------
        int
            Always `0`, since every basis direction carries a (finite) penalty under the GP prior.
        """
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        """Return the sum-to-zero constraint row for the intercept.

        Returns
        -------
        NDArray
            A `(1, k)` matrix whose product with the coefficient vector is zero when the smooth has
            mean zero over the training data.
        """
        self._check_fitted()
        B_train = self.basis_matrix(self._x_train)
        return B_train.mean(axis=0, keepdims=True)

    @property
    def n_basis(self) -> int:
        """Number of basis functions retained by this Gaussian process basis.

        Equal to the number of leading eigenfunctions of the covariance matrix kept
        during `fit()`. This is the requested `k` unless `fit()` was called with
        fewer training points than `k`, in which case it is silently reduced to `n`.

        Returns
        -------
        int
            The basis dimension `k`.
        """
        return self._k

    def __repr__(self) -> str:
        return f"GaussianProcess(k={self._k}, cov={self._cov!r})"
