# NetworkNet

**NetworkNet: A Deep Neural Network Approach for Random Networks with Sparse Nodal Attributes and Complex Nodal Heterogeneity**

This repository contains the official implementation of **NetworkNet**, a deep learning-based statistical framework designed for modeling random networks characterized by sparse nodal attributes and complex, high-dimensional nodal heterogeneity.

## Overview

NetworkNet addresses the challenges of traditional statistical network models when dealing with large-scale data where:
- Nodal features are high-dimensional and sparse.
- Nodal heterogeneity follows complex, non-linear patterns.
- Inter-nodal interactions are governed by latent structures that are difficult to specify manually.

The proposed DNN-based approach provides an end-to-end framework for network representation, link prediction, and inference while maintaining statistical rigor.

## Project Structure

The repository is organized as follows:

```text
NetworkNet/
├── data/               # Raw and processed datasets used in the study
├── simulation/         # Scripts for reproducing simulation experiments
│   ├── main_sim.py     # Main entry point for simulations
│   ├── models.py       # DNN architecture definitions (NetworkNet)
│   └── utils.py        # Helper functions for network generation and evaluation
├── empirical/          # Scripts for real-world data analysis
│   ├── analysis.py     # Empirical study implementation
│   └── plotting.py     # Visualization tools for results
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```


## Getting Started

- Python 3.8 or higher

- PyTorch (Recommended version 1.10+)

## Usage

1. Simulation Experiments: To run the simulation study and evaluate the model's performance under various sparsity levels and nodal heterogeneity settings
2. Empirical Analysis: To replicate the real-world data analysis (including the results shown in the manuscript)


## Citation
If you use this code or methodology in your research, please cite our paper:

@article{xing2026networknet,
  title={NetworkNet: A Deep Neural Network Approach for Random Networks with Sparse Nodal Attributes and Complex Nodal Heterogeneity},
  author={Xing, Zhaoyu and others},
  journal={Working Paper},
  year={2026}
}

## Contact

Zhaoyu Xing Robert and Sara Lumpkins Postdoctoral Research Associate

Department of Applied and Computational Mathematics and Statistics

University of Notre Dame

Email: zxing@nd.edu





