This repository contains code to reproduce the results of the RECOMB 2026 submission "Gaining mechanistic insight from geometric deep learning on molecule structures through equivariant convolution".

Installation: The package depends on torch, torch-scatter, and torch-geometric. Please follow their instructions for installation. Additionally, make sure to install the dependencies from https://github.com/BorgwardtLab/Cosmo and https://github.com/BorgwardtLab/bioverse.

Usage: Models can be trained with `bioverse train config.yaml exp=[mnist/beta2d/qm9aph/qm9cv0/qm9gap/qm9hom/qm9lum/qm9mu0]`. The figures of the paper can be reproduced with `python plot.py exp=[mnist/beta2d]`.