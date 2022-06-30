### Restructuring code for scikit-learn compatibility
1. Start with low-rank estimators.
   1. keywords: svd_solver = [full, randomized, arpack], backend = [keops, numpy]
   2. how to manage different kernels? sikit-learn accept "precomputed", we don't like it. Kernel should be an initialized kernel object.
2. The additional attributes such as eigenvalues, eigenvectors, modes should _always_ be computed?
3. Methods to implement are fit, predict, score.