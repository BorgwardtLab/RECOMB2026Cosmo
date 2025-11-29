This repository contains code to reproduce the results of the RECOMB 2026 submission "Gaining mechanistic insight from geometric deep learning on molecule structures through equivariant convolution".

Installation: `git clone` and `cd` into this repository. The package depends on the [bioverse](https://github.com/BorgwardtLab/bioverse) for data processing and evaluation pipelines, and on [cosmic-torch](https://github.com/BorgwardtLab/Cosmo) which contains the Cosmo layers. Install them with `pip install -r requirements.txt`. Make sure to before install [torch](https://pytorch.org/get-started/locally/) and [torch-scatter](https://pypi.org/project/torch-scatter/) according to their instructions and your system.

Usage: Models can be trained or tested with `bioverse [train/test] config.yaml exp=[mnist/beta2d/qm9aph/qm9cv0/qm9gap/qm9hom/qm9lum/qm9mu0]`. The figures of the paper can be reproduced with `python plot.py exp=[mnist/beta2d]`.

License: TBD