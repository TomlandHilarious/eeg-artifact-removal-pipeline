# Eye-Blink Artifact Removal from Single-Channel EEG with k-means and SSA

This repository contains a **PyTorch-based** reproduction of the paper:

> **Eye‑blink artifact removal from single-channel EEG with k‑means and SSA**  
> [*Scientific Reports (2021)*](https://doi.org/10.1038/s41598-021-90437-7)

The goal of this code is to **automate eye-blink artifact detection and removal** in single-channel EEG by combining:

- **k-Means clustering** for feature-based artifact detection  
- **Singular Spectrum Analysis (SSA)** to reconstruct and remove blink components

---

## Key Features

- Fully in **PyTorch** for easy tensor operations and potential GPU acceleration.  
- Clustering step implemented via **scikit-learn**’s `KMeans`.  
- Example code showing how to extract time-domain features (energy, mobility, kurtosis, etc.), apply k-means, and perform SSA-based reconstruction.

---

## Requirements

- **Python 3.6** (due to development version constraints)
- **PyTorch** (e.g., `1.x` compatible with Python 3.6)
- **numpy** (downgraded to a `1.x` version compatible with the PyTorch you use, e.g. `1.19.x`)
- **scikit-learn** (for the k-means clustering)

A sample `requirements.txt` might look like this:

torch==1.10.0 numpy==1.19.5 scikit-learn==0.24.2

