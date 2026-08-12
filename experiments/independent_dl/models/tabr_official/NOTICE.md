# TabR source notice

The retrieval, encoder–retrieval–predictor structure, label embedding, and
self-neighbor masking in `../tabr.py` are adapted from the official
**TabR: Tabular Deep Learning Meets Nearest Neighbors** implementation:

- Upstream: https://github.com/yandex-research/tabular-dl-tabr
- Pinned commit: `17baa9082506f8e7a0f8d11bb1e08212926a1507`
- Upstream file: `bin/tabr.py`
- License: MIT

The local adapter removes the upstream experiment harness and FAISS dependency.
It performs exact chunked PyTorch top-k search over only the current fold's
training rows.
