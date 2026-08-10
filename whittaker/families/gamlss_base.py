"""Abstract base class for GAMLSS distributional families."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray


class GAMLSSFamily(ABC):
    """Base class for GAMLSS distributional families.

    A GAMLSS family defines the full response distribution with multiple parameters (e.g. location
    and scale), each modelled by its own additive predictor. Subclasses must implement all abstract
    methods.
    """

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Names of distributional parameters, e.g. `('mu', 'sigma')`."""
        ...

    @abstractmethod
    def link(self, param: str, values: NDArray) -> NDArray:
        """Apply the link function for *param*: eta = g(theta)."""
        ...

    @abstractmethod
    def link_inverse(self, param: str, eta: NDArray) -> NDArray:
        """Apply the inverse link for *param*: theta = g^{-1}(eta)."""
        ...

    @abstractmethod
    def link_derivative(self, param: str, values: NDArray) -> NDArray:
        """Derivative of the link for *param*: d(eta)/d(theta)."""
        ...

    @abstractmethod
    def dl_dtheta(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """First derivative of the log-likelihood w.r.t. *param*."""
        ...

    @abstractmethod
    def d2l_dtheta2(self, param: str, y: NDArray, params: dict[str, NDArray]) -> NDArray:
        """Negative expected second derivative of the log-likelihood w.r.t. *param*.

        Must return positive values (the expected Fisher information diagonal for this parameter).
        """
        ...

    @abstractmethod
    def log_likelihood(self, y: NDArray, params: dict[str, NDArray]) -> float:
        """Full log-likelihood evaluated at the given parameter values."""
        ...

    @abstractmethod
    def initialize(self, y: NDArray) -> dict[str, NDArray]:
        """Starting values for all distributional parameters given *y*."""
        ...

    @abstractmethod
    def simulate(self, params: dict[str, NDArray], rng: object) -> NDArray:
        """Simulate response values from the distribution."""
        ...
