
import torch

"""
This module provides the core functions for performing Singular Spectrum Analysis (SSA) on 1D signals.

It implements the four classical steps of SSA:

1) Embedding (see `embedding`): 
   Converts a 1D signal into a 2D trajectory matrix by sliding a window over the signal.

2) Decomposition (see `decomposition`):
   Performs SVD on the trajectory matrix to obtain rank-1 components.

3) Grouping (see `grouping`):
   Uses eigenvalue ratios (sigma_i^2) to select which rank-1 components to keep.

4) Diagonal Averaging (see `diagonal_average`):
   Reconstructs a 1D signal from the (optionally) grouped trajectory matrix by averaging along diagonals.

These functions can be used together or individually
to analyze, separate, or denoise signals through the SSA framework.
"""

def embedding(signal:torch.Tensor, window_size=256):
    """
    Convert a 1D signal into a 2D 'trajectory matrix' using overlap (step=1).
    This is the first step in Singular Spectrum Analysis (SSA), where each
    column represents a shifted window of the original signal.

    Parameters
    ----------
    signal : array-like or torch.Tensor
        The 1D input signal (e.g., EEG) of length N.
    window_size : int, optional
        The length of each sliding window (default=256). Must be <= N.

    Returns
    -------
    X : torch.Tensor
        A 2D tensor (shape: [window_size, K]) where K = N - window_size + 1.
        Each column is one windowed segment of 'signal', shifted by 1 sample
        relative to the previous column.
    """
    x = torch.as_tensor(signal, dtype=torch.float32)
    N = x.shape[0]
    X = x.unfold(dimension=0, size=window_size, step=1).T
    print("Embedded matrix shape is", X.shape)
    return X

def decomposition(A_hat:torch.Tensor):
    """
    Perform the decomposition step of Singular Spectrum Analysis (SSA)
    on the trajectory matrix A_hat via SVD. 
    This is the first step in Singular Spectrum Analysis (SSA),

    where the rank-1 components A_i are formed as:
        A_i = sigma_i * (u_i outer v_i).

    Parameters
    ----------
    A_hat : torch.Tensor
        The trajectory matrix of shape (M, K) obtained from the embedding step.
        - M is the chosen window size (L in some texts).
        - K = N - M + 1, if N is the original signal length.
        Each column is a shifted window of the original 1D signal.

    Returns
    -------
    A_list : list of torch.Tensor
        A list of rank-1 matrices, each of shape (M, K). The length of this list
        is r = min(M, K). Each element A_i = sigma_i * (u_i outer v_i).
    S : torch.Tensor
        The singular values (sigma_i), of length r. If you square them,
        you get the eigenvalues (lambda_i = sigma_i^2).
    """
    U, S, Vt = torch.linalg.svd(A_hat, full_matrices=False)
    A_list = []
    for i in range(S.shape[0]):
        sigma_i = S[i]
        u_i = U[:, i]
        v_i = Vt[i,:]
        A_i = sigma_i * (u_i.unsqueeze(1) @ v_i.unsqueeze(0))
        A_list.append(A_i)
    
    return A_list, S

def grouping(A_list, singular_values, threshold=0.01):
    """
    Group or select rank-1 trajectory matrices based on their eigenvalue ratios,
    and sum the 'important' ones to form the final trajectory matrix.

    This corresponds to the third step in SSA, where each component's
    eigenvalue (lambda_i = sigma_i^2) is compared to the total variance.

    Parameters
    ----------
    A_list : list[torch.Tensor]
        A list of rank-1 trajectory matrices (A_i), each of shape (M, K).
        Typically obtained from ssa_decomposition. 
    singular_values : torch.Tensor
        The singular values (sigma_i), same length as A_list. 
    threshold : float, optional
        The fraction threshold T_SSA. A component i is kept if 
        (sigma_i^2 / sum_j(sigma_j^2)) > threshold. Default is 0.01.

    Returns
    -------
    A_sum : torch.Tensor
        The sum of all rank-1 matrices that pass the threshold criterion.
        Has shape (M, K), the same as each A_i.
    kept_indices : list of int
        The indices i of the components that were retained. Useful for debugging
        or analysis.
    """
    sigmas = singular_values.float()
    lambdas = sigmas**2
    total_lambda = torch.sum(lambdas)

    if total_lambda <= 0:
        return torch.zeros_like(A_list[0]), []
    
    ratios = lambdas / total_lambda
    kept_indices = []
    for i in range(len(A_list)):
        if ratios[i].item() > threshold:
            kept_indices.append(i)  
    if len(kept_indices) == 0:
        A_sum = torch.zeros_like(A_list[0])
    else:
        A_sum = A_list[kept_indices[0]].clone()
        for idx in kept_indices[1:]:
            A_sum += A_list[idx]
    return A_sum, kept_indices

def diagonal_average(X_bar: torch.Tensor) -> torch.Tensor:
    """
    Diagonal-average (Hankelize) the M x K matrix X_bar into a 1D signal s
    of length (M + K - 1). This is the fourth step in SSA.
    
    Parameters
    ----------
    X_bar : torch.Tensor
        A 2D tensor of shape (M, K) representing the partial or grouped
        trajectory matrix in SSA. M is the window size, and K is the number
        of columns
    
    Returns
    -------
    s : torch.Tensor
        A 1D tensor of length (M + K - 1), obtained by averaging along the
        diagonals (r + c = const) of X_bar.
    """
    M, K = X_bar.shape
    N = M + K - 1
    # prepare tensors
    s = torch.zeros(N, dtype=X_bar.dtype, device=X_bar.device)
    
    for n in range(1, N + 1):
        if n < M:
            elems = [X_bar[i, n-i-1] for i in range(n)]
            s[n-1] = torch.mean(torch.stack(elems))
        elif n <= K:
            elems = [X_bar[i, n-i-1] for i in range(M)]
            s[n-1] = torch.mean(torch.stack(elems))
        else:
            elems = [X_bar[i, n-i-1] for i in range(n-K, M)]
            s[n-1] = torch.mean(torch.stack(elems))

    return s


def diagonal_average_vectorized(X_bar: torch.Tensor) -> torch.Tensor:
    """
    Vectorized implementation of diagonal averaging (Hankelization) for an M x K matrix X_bar.
    Returns a 1D tensor s of length (M + K - 1), where each element is the average of elements
    along the corresponding anti-diagonal (i + j = constant).

    Parameters
    ----------
    X_bar : torch.Tensor
        A 2D tensor of shape (M, K).

    Returns
    -------
    s : torch.Tensor
        A 1D tensor of length (M + K - 1) containing the diagonal averages.
    """
    M, K = X_bar.shape
    N = M + K - 1

    # Create index matrices for rows and columns
    row_idx = torch.arange(M, device=X_bar.device).view(M, 1).expand(M, K)
    col_idx = torch.arange(K, device=X_bar.device).view(1, K).expand(M, K)
    
    # Each element at (i, j) belongs to diagonal with index i+j
    diag_idx = row_idx + col_idx  # shape: (M, K)
    
    # Flatten X_bar and diag_idx into 1D tensors
    X_flat = X_bar.reshape(-1)
    diag_idx_flat = diag_idx.reshape(-1)
    
    # Prepare tensors to accumulate sums and counts for each diagonal
    sums = torch.zeros(N, dtype=X_bar.dtype, device=X_bar.device)
    counts = torch.zeros(N, dtype=X_bar.dtype, device=X_bar.device)
    
    # scatter_add: add X_flat's values into sums at positions given by diag_idx_flat
    sums.scatter_add_(0, diag_idx_flat, X_flat)
    # Also count how many elements are summed per diagonal
    counts.scatter_add_(0, diag_idx_flat, torch.ones_like(X_flat))
    
    # Compute the average for each diagonal
    s = sums / counts
    return s




