import torch
"""
This module provides time-domain feature extraction functions for EEG signals.

It includes:

- compute_energy(x): Returns the sum of squares within a signal segment.
- hjorth_mobility(x): Calculates Hjorth Mobility, measuring how quickly the signal changes.
- kurtosis(x): Either raw or excess kurtosis, quantifying how heavy-tailed the distribution is.
- peak_to_peak(x): The difference between the maximum and minimum (absolute) amplitudes.
- extract_features(ssa_embedding): Applies the above four features column-by-column to an SSA-embedded matrix.

All of these functions can be used individually or within a pipeline to characterize EEG windows
for classification, artifact detection, or other signal processing tasks.
"""

def compute_energy(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the energy feature for a given 1D signal segment.
    Energy is defined as the sum of squares of the amplitudes.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor representing the signal segment.

    Returns
    -------
    torch.Tensor
        A scalar (0D tensor) with the sum of squares of x.
    """
    return torch.sum(x**2)

def hjorth_mobility(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Hjorth Mobility feature.

    Hjorth Mobility = sqrt(var(dx) / var(x)),
    where dx is the first difference of x.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor representing the signal segment.

    Returns
    -------
    torch.Tensor
        A scalar (0D tensor) representing the mobility.
    """
    
    dx = torch.diff(x)
    var_x = torch.var(x, unbiased=True)
    var_dx = torch.var(dx, unbiased=True)
    if var_x == 0:
        return 0.0
    mobility = torch.sqrt(var_dx / var_x)
    return mobility

def kurtosis(x: torch.Tensor, excess: bool = False) -> torch.Tensor:
    """
    Compute kurtosis of a 1D signal segment.
    Optionally compute "excess" kurtosis (subtract 3) if excess=True.

    Parameters
    ----------
    x : torch.Tensor
        1D tensor representing the signal segment.
    excess : bool, optional
        If True, compute excess kurtosis (raw kurtosis - 3). Default is False.

    Returns
    -------
    torch.Tensor
        A scalar (0D tensor) for the kurtosis.
    """
    x = x.to(dtype=torch.float32)
    mean_x = torch.mean(x)
    var_x = torch.var(x, unbiased=True)

    if var_x == 0:
        return torch.tensor(0.0)
    
    fourth_moment = torch.mean((x - mean_x) **4)
    raw_kurtosis = fourth_moment / (var_x**2)
    
    if excess:
        return raw_kurtosis - 3
    else:
        return raw_kurtosis
    
def peak_to_peak(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the peak-to-peak amplitude of a 1D signal segment.
    Defined as (max(x) - abs(min(x))).

    Parameters
    ----------
    x : torch.Tensor
        1D tensor representing the signal segment.

    Returns
    -------
    torch.Tensor
        A scalar (0D tensor) for the peak-to-peak amplitude.
    """
    return torch.max(x) - abs(torch.min(x))


def extract_features(ssa_embedding: torch.Tensor) -> torch.Tensor:
    """
    Extract four time-domain features from each column of an SSA-embedded matrix.

    The matrix 'ssa_embedding' is assumed to have shape (L, K),
    where L is the window size and K is the number of overlapping segments.
    For each column in (L,), we compute:
      1) energy (sum of squares)
      2) Hjorth mobility
      3) kurtosis
      4) peak-to-peak amplitude

    Parameters
    ----------
    ssa_embedding : torch.Tensor
        2D tensor of shape (L, K), where each column is a signal window.

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape (4, K) containing the four features for each column.
    """

    L, K = ssa_embedding.shape
    features = torch.zeros((4, K), dtype=torch.float32)
    for k in range(K):
        col = ssa_embedding[:, k]
        e = compute_energy(col)
        m = hjorth_mobility(col)
        ku = kurtosis(col)
        p2p = peak_to_peak(col)

        features[0, k] = e
        features[1, k] = m
        features[2, k] = ku
        features[3, k] = p2p
    
    return features