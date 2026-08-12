"""Soap film smooth for 2-D domains with complex boundaries.

Implements the soap-film smoother of Wood, Bravington & Hedley (2008). The smooth is decomposed into
a boundary component (1-D cyclic spline around the boundary) and an interior component (PDE-based
finite-element solution). This allows smoothing over irregular 2-D regions with holes.

The boundary is given as a list of 2-D polygons (the first is the outer boundary; any additional
polygons are holes). Internally the domain is triangulated and a finite-element discretisation of
the thin-plate energy is used.

Usage in a formula:

    s(x, y, bs="so", xt={"boundary": bnd, "knots": knots})

where `bnd` is a list of (m, 2) arrays tracing each boundary loop and `knots` is an (nk, 2) array of
interior knot locations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

from whittaker.smooths.base import SmoothBasis

_EPS = np.finfo(float).eps


def _point_in_polygon(px: float, py: float, poly: NDArray) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    x1, y1 = poly[0]
    for i in range(1, n + 1):
        x2, y2 = poly[i % n]
        if min(y1, y2) < py <= max(y1, y2):
            if px <= max(x1, x2):
                if y1 != y2:
                    xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                if y1 == y2 or px <= xinters:
                    inside = not inside
        x1, y1 = x2, y2
    return inside


def _points_in_domain(pts: NDArray, boundary: list[NDArray]) -> NDArray:
    """Test which points lie inside the domain (outer boundary minus holes)."""
    mask = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        inside_outer = _point_in_polygon(pts[i, 0], pts[i, 1], boundary[0])
        if not inside_outer:
            continue
        in_hole = False
        for hole in boundary[1:]:
            if _point_in_polygon(pts[i, 0], pts[i, 1], hole):
                in_hole = True
                break
        mask[i] = not in_hole
    return mask


def _boundary_distance(pt: NDArray, boundary: list[NDArray]) -> float:
    """Minimum distance from a point to all boundary segments."""
    min_dist = np.inf
    for loop in boundary:
        n = len(loop)
        for i in range(n):
            a = loop[i]
            b = loop[(i + 1) % n]
            ab = b - a
            ap = pt - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + _EPS), 0, 1)
            closest = a + t * ab
            d = np.linalg.norm(pt - closest)
            if d < min_dist:
                min_dist = d
    return min_dist


def _fem_matrices(
    knots: NDArray,
    boundary_pts: NDArray,
    boundary: list[NDArray],
) -> tuple[NDArray, NDArray, Delaunay]:
    """Build finite-element stiffness and mass matrices on a triangulation.

    Returns (K, M, tri) where K is the stiffness matrix (integral of grad phi_i . grad phi_j) and M
    is the mass matrix (integral of phi_i * phi_j).
    """
    all_pts = np.vstack([knots, boundary_pts])
    n_interior = len(knots)

    tri = Delaunay(all_pts)

    n_pts = len(all_pts)
    K = np.zeros((n_pts, n_pts))
    M = np.zeros((n_pts, n_pts))

    for simplex in tri.simplices:
        i, j, k = simplex
        p = all_pts[[i, j, k]]

        d = np.array(
            [
                [p[1, 0] - p[2, 0], p[2, 0] - p[0, 0], p[0, 0] - p[1, 0]],
                [p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]],
            ]
        )

        area = 0.5 * abs(
            (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        )
        if area < _EPS:
            continue

        grad = np.array(
            [
                [p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]],
                [p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]],
            ]
        ) / (2.0 * area)

        K_local = area * (grad.T @ grad)

        M_local = area / 12.0 * (np.ones((3, 3)) + np.eye(3))

        for a in range(3):
            for b in range(3):
                K[simplex[a], simplex[b]] += K_local[a, b]
                M[simplex[a], simplex[b]] += M_local[a, b]

    return K[:n_interior, :n_interior], M[:n_interior, :n_interior], tri


class SoapFilm(SmoothBasis):
    r"""Soap film smooth for 2-D domains with complex boundaries.

    Implements the soap-film smoother of Wood, Bravington & Hedley (2008). Ordinary 2-D smooths
    such as `TPRS` or tensor-product splines treat the covariate domain as if it were convex and
    unobstructed; when the true domain has holes, concave coastlines, peninsulas, or other
    complicated shapes, such smooths will "leak" information across boundaries that are close in
    Euclidean distance but far apart when you have to travel around the domain (e.g. two points on
    opposite banks of a narrow bay). `SoapFilm` avoids this by finite-element-discretizing the
    domain itself: the smooth is represented as a piecewise-linear function on a triangulation of
    the actual (possibly non-convex, possibly multiply-connected) region, so its value at any point
    only depends on interior knots reachable through the domain. Use `SoapFilm` instead of a
    standard 2-D smooth whenever the covariates are genuinely spatial coordinates and the region
    they live in is not simply a rectangle — for example coastal, riverine, or other geographically
    constrained data.

    Parameters
    ----------
    boundary:
        List of boundary loops, each an `(m, 2)` array of ordered vertices tracing a closed polygon.
        The first loop is the outer boundary of the domain; any additional loops are holes cut out
        of it (e.g. islands or excluded regions). If not supplied, a padded rectangular bounding box
        around the training data is used, which reduces the smooth to an ordinary (simply-connected,
        convex) domain — supply an explicit boundary whenever the domain has a non-trivial shape.
    knots:
        Interior knot locations as an `(nk, 2)` array; these become the nodes of the finite-element
        triangulation and directly determine the basis dimension. If not supplied, a roughly
        square grid of candidate points is generated over the bounding box, filtered to those lying
        inside the domain (outer boundary minus holes), and then subsampled down to at most `k`
        points. Supplying knots explicitly gives more control over their placement, which matters
        near sharp domain features.
    k:
        Target number of basis functions. If `knots` is not supplied, this determines the density of
        the automatically generated knot grid, and the actual number of basis functions equals the
        number of interior knots retained (which may be less than `k`). If `knots` is supplied
        directly, `k` is not used to size the basis; the number of basis functions equals
        `len(knots)`. The default is `30`.

    Notes
    -----
    Fitting proceeds in three stages:

    1. **Triangulation.** Interior knots are combined with points sampled along the boundary loops
       (each boundary segment contributes its endpoint and midpoint) and a Delaunay triangulation is
       built over all of these points.
    2. **Finite-element assembly.** On each triangle, standard piecewise-linear (barycentric) basis
       functions `phi_i` are used to assemble a stiffness matrix `K` (with entries
       `K_{ij} = integral of grad(phi_i) . grad(phi_j)` over the domain) and a mass matrix `M`
       (with entries `M_{ij} = integral of phi_i * phi_j`), then both are restricted to the rows and
       columns corresponding to interior knots (boundary points are not free parameters).
    3. **Basis evaluation.** At an arbitrary evaluation point, the containing triangle is located
       (via `Delaunay.find_simplex`) and the point's barycentric coordinates within that triangle
       give its basis-function weights; points outside the triangulation fall back to a nearest-knot
       indicator.

    The penalty matrix is exactly the interior-restricted stiffness matrix,

    $$
    \mathbf{S} = \mathbf{K}_{\text{interior}}, \qquad
    \boldsymbol{\beta}^\top \mathbf{S} \boldsymbol{\beta} = \int_\Omega \lVert \nabla f \rVert^2 \, dA,
    $$

    the discretized Dirichlet energy (membrane/thin-film bending energy) of the fitted surface over
    the domain `Omega`, consistent with the "soap film" interpretation: the fitted surface behaves
    like a soap film stretched across the (possibly perforated) domain boundary. `S` is positive
    semi-definite; its null space corresponds to the constant function, so `null_space_dimension()`
    is `0` here by convention (the constant is handled via `identifiability_constraints()` rather
    than being excluded from the penalty). Because the basis is piecewise-linear on a triangulation
    rather than smooth in the classical sense, the fitted surface is continuous but only
    once-differentiable, and the quality of the fit is sensitive to the density and placement of
    interior knots relative to the sharpness of the domain's boundary features — very thin or
    highly concave regions may need denser knots near the constriction to avoid leakage.

    Examples
    --------
    ```{python}
    import numpy as np
    from whittaker.smooths import SoapFilm

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (100, 2))

    basis = SoapFilm(k=20).fit(x)
    B = basis.basis_matrix(x)
    S = basis.penalty_matrix()
    B.shape, S.shape
    ```
    """

    def __init__(
        self,
        *,
        boundary: list[NDArray] | None = None,
        knots: NDArray | None = None,
        k: int = 30,
    ) -> None:
        self._boundary = boundary
        self._knots = knots
        self._k = k
        self._fitted = False
        self._K: NDArray | None = None
        self._M: NDArray | None = None
        self._tri: Delaunay | None = None
        self._all_pts: NDArray | None = None
        self._n_interior: int = 0

    def fit(self, x: NDArray) -> SoapFilm:
        """Fit the soap film smooth to training data `x`.

        Determines the domain boundary and interior knots (if not already supplied), triangulates
        the domain, and assembles the finite-element stiffness and mass matrices.

        Parameters
        ----------
        x:
            Training covariates. Shape `(n, 2)`; `SoapFilm` only supports exactly two covariates
            (spatial x/y coordinates).

        Returns
        -------
        SoapFilm
            Returns `self` for method chaining.

        Raises
        ------
        ValueError
            If `x` does not have exactly 2 columns.
        """
        x = self._as_2d(x)
        if x.shape[1] != 2:
            raise ValueError("SoapFilm requires exactly 2 covariates.")

        if self._boundary is None:
            xmin, xmax = x[:, 0].min(), x[:, 0].max()
            ymin, ymax = x[:, 1].min(), x[:, 1].max()
            pad_x = (xmax - xmin) * 0.02 + _EPS
            pad_y = (ymax - ymin) * 0.02 + _EPS
            self._boundary = [
                np.array(
                    [
                        [xmin - pad_x, ymin - pad_y],
                        [xmax + pad_x, ymin - pad_y],
                        [xmax + pad_x, ymax + pad_y],
                        [xmin - pad_x, ymax + pad_y],
                    ]
                )
            ]

        if self._knots is None:
            n_side = max(int(np.sqrt(self._k)) + 1, 4)
            xmin, xmax = x[:, 0].min(), x[:, 0].max()
            ymin, ymax = x[:, 1].min(), x[:, 1].max()
            gx = np.linspace(xmin, xmax, n_side + 2)[1:-1]
            gy = np.linspace(ymin, ymax, n_side + 2)[1:-1]
            grid_x, grid_y = np.meshgrid(gx, gy)
            candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            inside = _points_in_domain(candidates, self._boundary)
            self._knots = candidates[inside]
            if len(self._knots) > self._k:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(self._knots), self._k, replace=False)
                self._knots = self._knots[np.sort(idx)]

        self._n_interior = len(self._knots)

        bnd_pts_list = []
        for loop in self._boundary:
            n_loop = len(loop)
            for i in range(n_loop):
                seg_start = loop[i]
                seg_end = loop[(i + 1) % n_loop]
                bnd_pts_list.append(seg_start)
                mid = (seg_start + seg_end) / 2
                bnd_pts_list.append(mid)
        boundary_pts = np.array(bnd_pts_list)

        from scipy.spatial import cKDTree

        tree = cKDTree(boundary_pts)
        _, idx = tree.query(boundary_pts)
        seen = set()
        unique_mask = []
        for i in range(len(boundary_pts)):
            key = tuple(np.round(boundary_pts[i], 10))
            if key not in seen:
                seen.add(key)
                unique_mask.append(True)
            else:
                unique_mask.append(False)
        boundary_pts = boundary_pts[unique_mask]

        self._K, self._M, self._tri = _fem_matrices(self._knots, boundary_pts, self._boundary)
        self._all_pts = np.vstack([self._knots, boundary_pts])

        self._fitted = True
        return self

    def basis_matrix(self, x: NDArray) -> NDArray:
        """Evaluate the soap film basis at `x`.

        For each point, locates the containing triangle of the fitted triangulation and returns its
        barycentric coordinates as basis-function weights against the interior knots; points that
        fall outside the triangulation (e.g. slightly outside the training domain) fall back to a
        one-hot indicator of the nearest interior knot.

        Parameters
        ----------
        x:
            Evaluation points. Shape `(n, 2)`.

        Returns
        -------
        NDArray
            Design matrix of shape `(n, k)` where `k` is the number of interior knots.
        """
        x = self._as_2d(x)
        n = x.shape[0]
        nk = self._n_interior

        B = np.zeros((n, nk))
        tri = self._tri
        all_pts = self._all_pts

        simplex_indices = tri.find_simplex(x)

        for i in range(n):
            si = simplex_indices[i]
            if si < 0:
                dists = np.linalg.norm(self._knots - x[i], axis=1)
                nearest = np.argmin(dists)
                B[i, nearest] = 1.0
                continue

            verts = tri.simplices[si]
            p = all_pts[verts]

            area_total = 0.5 * abs(
                (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
            )
            if area_total < _EPS:
                for v in verts:
                    if v < nk:
                        B[i, v] = 1.0 / max(sum(1 for vv in verts if vv < nk), 1)
                continue

            bary = np.zeros(3)
            for j in range(3):
                j1 = (j + 1) % 3
                j2 = (j + 2) % 3
                bary[j] = (
                    0.5
                    * abs(
                        (p[j1, 0] - x[i, 0]) * (p[j2, 1] - x[i, 1])
                        - (p[j2, 0] - x[i, 0]) * (p[j1, 1] - x[i, 1])
                    )
                    / area_total
                )

            for j in range(3):
                v = verts[j]
                if v < nk:
                    B[i, v] = bary[j]

        return B

    def penalty_matrix(self) -> NDArray:
        """Return the `k x k` penalty matrix `S`, the interior-restricted stiffness matrix.

        `S` is the finite-element discretization of the Dirichlet energy
        `integral of ||grad(f)||^2` over the domain, restricted to the interior knot degrees of
        freedom.

        Returns
        -------
        NDArray
            Shape `(k, k)`, positive semi-definite.
        """
        return self._K.copy()

    def null_space_dimension(self) -> int:
        """Return `0` by convention for the soap film penalty.

        The stiffness matrix's true null space is spanned by the constant function, but
        `SoapFilm` handles the constant via `identifiability_constraints()` rather than excluding
        it from the penalty, so this method reports `0`.

        Returns
        -------
        int
            Always `0`.
        """
        return 0

    def identifiability_constraints(self) -> NDArray | None:
        """Return the mean-constraint row for the intercept.

        Returns
        -------
        NDArray
            A `(1, k)` matrix (uniform weights `1 / k`) whose product with the coefficient vector
            is zero when the smooth has mean zero over the interior knots.
        """
        return np.ones((1, self._n_interior)) / self._n_interior

    @property
    def n_basis(self) -> int:
        """Number of basis functions, equal to the number of interior knots."""
        return self._n_interior

    def __repr__(self) -> str:
        return f"SoapFilm(k={self._n_interior})"
