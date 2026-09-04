"""Gaussian-kernel GMDC objective for TESR representation learning."""

from __future__ import annotations

from typing import Union

import torch
from torch import nn


Bandwidth = Union[float, str]


def _as_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    """Convert a batched tensor to ``(batch, flattened_features)`` form."""
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if value.ndim < 2:
        raise ValueError(f"{name} must contain a batch dimension")
    return value.flatten(start_dim=1)


def _pairwise_squared_distances(value: torch.Tensor) -> torch.Tensor:
    differences = value.unsqueeze(1) - value.unsqueeze(0)
    return differences.square().sum(dim=-1).clamp_min(0)


def _median_bandwidth_squared(distances_squared: torch.Tensor) -> torch.Tensor:
    """Return a detached, robust median-heuristic value for sigma squared."""
    positive = distances_squared[distances_squared > 0]
    if positive.numel() == 0:
        return distances_squared.new_tensor(1.0)
    epsilon = torch.finfo(distances_squared.dtype).eps
    return positive.median().detach().clamp_min(epsilon)


def gaussian_kernel(
    response: torch.Tensor, bandwidth: Bandwidth = "median"
) -> torch.Tensor:
    """Build ``exp(-||y_i-y_j||^2 / (2 sigma^2))`` for one mini-batch.

    ``bandwidth="median"`` sets ``sigma^2`` to the median strictly positive
    pairwise squared response distance.  Ignoring zero distances makes the
    heuristic usable for responses with repeated values, including class
    labels.  When all responses coincide the fallback bandwidth is one and
    the resulting all-ones kernel correctly gives zero GMDC.
    """
    response = _as_matrix(response, "response")
    if not response.is_floating_point():
        response = response.float()
    distances_squared = _pairwise_squared_distances(response)

    if bandwidth == "median":
        bandwidth_squared = _median_bandwidth_squared(distances_squared)
    elif isinstance(bandwidth, (int, float)) and not isinstance(bandwidth, bool):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be a positive number or 'median'")
        bandwidth_squared = distances_squared.new_tensor(float(bandwidth) ** 2)
    else:
        raise ValueError("bandwidth must be a positive number or 'median'")

    return torch.exp(-distances_squared / (2 * bandwidth_squared))


def _double_center(matrix: torch.Tensor) -> torch.Tensor:
    """Center a square Gram matrix in feature space."""
    return (
        matrix
        - matrix.mean(dim=0, keepdim=True)
        - matrix.mean(dim=1, keepdim=True)
        + matrix.mean()
    )


class Loss_GMDC(nn.Module):
    """Normalized Gaussian-kernel generalized martingale difference score.

    The raw Gaussian-kernel GMDD is the Frobenius inner product between the
    centered linear Gram matrix of a learned representation ``U`` and the
    centered Gaussian Gram matrix of a response ``V``.  GMDC divides this
    quantity by both Gram matrices' Frobenius norms:

        <G_U, G_V> / sqrt(<G_U, G_U> <G_V, G_V>).

    This uses the squared-correlation convention of TESR's ``Loss_DC`` and
    returns a scale-free score in ``[0, 1]``.  Larger values mean stronger
    conditional-mean dependence.  Consequently TESR should subtract this
    score when minimizing its generator objective.
    """

    def __init__(self, bandwidth: Bandwidth = "median") -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def gmdc(self, representation: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        representation = _as_matrix(representation, "representation")
        response = _as_matrix(response, "response")

        if representation.shape[0] != response.shape[0]:
            raise ValueError("representation and response must have the same batch size")
        if not representation.is_floating_point():
            raise TypeError("representation must be a floating-point tensor")
        if representation.shape[0] == 0:
            raise ValueError("GMDC requires a non-empty batch")
        if representation.shape[0] == 1:
            # A dependence coefficient is undefined for one observation.  TESR
            # can create singleton source groups, so match its historical DC
            # behavior with a neutral score that remains connected to autograd.
            return representation.sum() * 0.0

        response = response.to(
            device=representation.device, dtype=representation.dtype
        )
        centered = representation - representation.mean(dim=0, keepdim=True)
        feature_gram = centered @ centered.transpose(0, 1)
        response_kernel = gaussian_kernel(response, bandwidth=self.bandwidth)
        centered_kernel = _double_center(response_kernel)

        numerator = (feature_gram * centered_kernel).mean()
        representation_scale = feature_gram.square().mean()
        response_scale = centered_kernel.square().mean()
        epsilon = torch.finfo(representation.dtype).eps
        scale_product = (representation_scale * response_scale).clamp_min(epsilon)
        denominator = torch.sqrt(scale_product)
        score = numerator / denominator

        # Centered positive-semidefinite Gram matrices have a non-negative
        # Frobenius inner product; clamping only removes round-off excursions.
        return score.clamp(min=0.0, max=1.0)

    def forward(
        self, representation: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        return self.gmdc(representation, response)


__all__ = ["Bandwidth", "Loss_GMDC", "gaussian_kernel"]
