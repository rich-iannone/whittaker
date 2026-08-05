"""Abstract base class for response distribution families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Family(ABC):
    """Abstract family defining the response distribution and link function.

    A family encapsulates three things:

    1. The relationship between the mean μ and the linear predictor η (the link).
    2. The variance function V(μ) relating the variance to the mean.
    3. Deviance and log-likelihood computations.

    Subclasses must implement all abstract methods. The link function and its
    inverse/derivative are used by the P-IRLS algorithm to form pseudo-data
    and working weights.
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
    def deviance(self, y: NDArray, mu: NDArray) -> float:
        """Total (unscaled) deviance: 2 * Σ [ℓ(y; y) − ℓ(y; μ)]."""
        ...

    @abstractmethod
    def log_likelihood(self, y: NDArray, mu: NDArray, scale: float) -> float:
        """Log-likelihood ℓ(y; μ, φ) evaluated at the given scale parameter φ."""
        ...

    @property
    def scale_known(self) -> bool:
        """Whether the scale parameter is fixed (True for Binomial, Poisson)."""
        return False

    def initialize(self, y: NDArray) -> NDArray:
        """Starting values for μ given the response *y*.

        The default returns *y* unchanged, which is appropriate for Gaussian with identity link.
        Families with non-identity links or constrained means should override this.
        """
        return y.copy()
