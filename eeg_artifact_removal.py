import torch
import numpy as np
from sklearn.cluster import KMeans
from feature import extract_features
from ssa import embedding, decomposition, grouping, diagonal_average
from typing import List
import time as pytime
print(torch.__version__)

def k_means_clustering(feature_matrix: torch.Tensor, num_clusters: int):
    """
    Step 3: Perform k-means clustering on feature dimension

    This function uses scikit-learn's KMeans to group the feature vectors
    into 'num_clusters' clusters. The rows of 'feature_matrix' represent
    features, while the columns represent different
    segments/windows. 

    Parameters
    ----------
    feature_matrix : torch.Tensor
        2D tensor of shape (num_features, num_segments). 
    num_clusters : int
        The desired number of clusters.

    Returns
    -------
    labels : np.ndarray
        A 1D array of shape (num_segments,) containing the cluster label
        for each column in 'feature_matrix'.
    centers : np.ndarray
        A 2D array of shape (num_clusters, num_features), representing
        the cluster centroids in feature space.
    """
    feature_matrix_np = feature_matrix.T.cpu().numpy().astype(np.float32)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(feature_matrix_np)
    # Centroids: shape (num_clusters, num_features)
    centers = kmeans.cluster_centers_
    return labels, centers


def reconstruct_signals(X:torch.Tensor, labels:np.ndarray, num_clusters:int):

    L, K = X.shape
    def create_cluster_matrix(X:torch.Tensor, cluster_idx:int):
        """
        Creates the cluster-specific matrix X̄ᵢ (shape L x K)
        by copying columns from X if labels[j] == cluster_idx,
        else putting 0 in that column.
        """
        X_i = torch.zeros_like(X)
        for j in range(K):
            if labels[j] == cluster_idx:
                X_i[:, j] = X[:, j]
        return X_i  
    # Create and reconstruct each cluster
    signals = []
    for cluster_idx in range(num_clusters):
        # 1) Build X̄ᵢ
        X_i = create_cluster_matrix(X, cluster_idx)
        # 2) Diagonal-average => 1D signal
        s_i = diagonal_average(X_i)
        signals.append(s_i)

    return signals

def fractal_sevcik(signal:torch.Tensor) -> float:
    """
    Sevcik Fractal Dimension (SFD) referred by the original paper, 
        adapted from 
        https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/complexity/fractal_sevcik.py
    """
    n = signal.shape[0]
    s_min = torch.min(signal)
    s_max = torch.max(signal)

    # 1) Normalize the signal (new range to [0, 1])
    y_ = (signal - s_min) / (s_max - s_min)
    # 2) Derive x* and y* (y* is actually the normalized signal)
    x_ = torch.linspace(0, 1, steps=n, dtype=torch.float32, device=signal.device)
    # 3) Compute L (because we use np.diff, hence n-1 below)
    dy = y_[1:] - y_[:-1]
    dx = x_[1:] - x_[:-1]
    dist = torch.sqrt(dx**2 + dy**2)
    L = torch.sum(dist)
    # 4） Compute the fractal dimension (approximation)
    sfd = 1.0 + torch.log(L) / torch.log(torch.tensor(2.0 * (n - 1), device=signal.device))
    return float(sfd.item())

def refine_artifact_with_ssa(
        artifact_signal: torch.Tensor,
        window_size: int,
        grouping_threshold: float):
    """
    Apply SSA to the artifact_signal to remove EEG remnants.
    """
    artifact_signal = embedding(artifact_signal, window_size)
    decomposed_artifact_signals, singular_values = decomposition(artifact_signal)
    A_sum, _ = grouping(decomposed_artifact_signals, singular_values, grouping_threshold)
    blink_artifact = diagonal_average(A_sum)
    return blink_artifact

def remove_blink_artifact(cluster_signals:List[torch.Tensor],
                          window_size:int,
                          eeg_signal: torch.Tensor,
                          fd_threshold:float,
                          grouping_threshold:float):
    """
    Identify and remove eye-blink artifact from EEG by:
      1) Computing FD for each cluster signal,
      2) Selecting those with FD <= fd_threshold (blink-like),
      3) Summing those to form the blink estimate,
      4) Subtracting from the original EEG.

    Parameters
    ----------
    cluster_signals : List[torch.Tensor]
        A list of length L, where each element is a 1D PyTorch tensor
        of the same length as eeg_signal. These are the signals reconstructed
        from Step 4 (i.e., from diagonal averaging of each cluster).
    eeg_signal : torch.Tensor
        The original 1D EEG recording (contaminated by blink).
    fd_threshold : float
        The preset fractal dimension threshold. Any signal with FD <= this
        is considered blink-artifact.

    Returns
    -------
    cleaned_eeg : torch.Tensor
        The artifact-corrected EEG (same shape as eeg_signal).
    blink_estimate : torch.Tensor
        The summed blink-artifact signal (same shape as eeg_signal).
    fd_values : list[float]
        The FD values for each cluster signal, in the same order as cluster_signals.
    """
    #  Compute FD & sum blink components
    eeg_signal = eeg_signal.flatten()
    fd_values = []
    blink_components = []
    for signal in cluster_signals:
        fd_val = fractal_sevcik(signal)
        fd_values.append(fd_val)
        if fd_val <= fd_threshold:
            blink_components.append(signal)
    if len(blink_components) == 0:
        a_r = torch.zeros_like(eeg_signal)
    else:
        a_r = blink_components[0].clone()
        for b_idx in range(1, len(blink_components)):
            a_r += blink_components[b_idx]

    # Convert a_r to a binary template a_b (nonzero -> 1, zero -> 0)
    a_b = (a_r != 0).float()
    # Multiply the binary mask by the original EEG x
    #         => blink_artifact = a_b * x
    blink_artifact = a_b * eeg_signal
    # refine the artifact with SSA
    blink_artifact = refine_artifact_with_ssa(blink_artifact, window_size, grouping_threshold)
    # subtract the artifact from the contaminated signal
    cleaned_eeg = eeg_signal - blink_artifact
    
    return cleaned_eeg, blink_artifact, fd_values


def artfiact_removal_pipeline(contaminated_eeg: torch.tensor,
                              window_size: int,
                              num_clusters: int,
                              fd_threshold: float,
                              grouping_threshold: float):
    """
    The function that executes the complete eyeblink artifact removal pipeline.
    """
    # Step 1: embed the contaminated signal using SSA embedding step
    X = embedding(contaminated_eeg, window_size)
    # Step 2: feature extraction from the raw signal
    feature_matrix = extract_features(X)
    # Step 3: performs k-means clustering on the feature space
    labels, _ = k_means_clustering(feature_matrix, num_clusters)
    # Step 4: reconstruct the signals from the clustering reults
    cluster_signals = reconstruct_signals(X, labels, num_clusters)
    # Step 5 - 9: Identify and refine the eye-blink artifact
    cleaned_eeg, blink_est, fd_vals = remove_blink_artifact(cluster_signals,
                                                            window_size,
                                                            contaminated_eeg,
                                                            fd_threshold,
                                                            grouping_threshold)
    return cleaned_eeg, blink_est, fd_vals

if __name__ == "__main__":
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'
    # synthetic data
    torch.manual_seed(42)
    n_samples = 50000
    sample_rate = 256
    time = torch.arange(0, n_samples) / sample_rate
    base_signal = 0.1 * torch.randn(n_samples)
    blink_indices = torch.randint(0, n_samples, size = (5,), device=device)
    for idx in blink_indices:
        start = idx
        end = min(idx + 50, n_samples)
        base_signal[start:end] += torch.linspace(3, 0, end - start, device=device)
    eeg_signal = base_signal

    # function arguments
    # parameter control
    # best performance parameters provided below
    window_size = 128
    num_clusters = 4
    fd_threshold = 1.4
    grouping_threshold = 0.01


    start_time = pytime.time()


    cleaned_eeg, _, _ = artfiact_removal_pipeline(contaminated_eeg=eeg_signal, 
                              window_size=window_size,
                              num_clusters=num_clusters,
                              fd_threshold=fd_threshold,
                              grouping_threshold=grouping_threshold
                              )
    print(cleaned_eeg)
    end_time = pytime.time()
    print("processing done!")
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")