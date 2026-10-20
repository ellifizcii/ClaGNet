CLaGNet — Contrastive Latent Gaussian Network for Open-Set Cell Type Annotation
Research code for CLaGNet, an open-set cell type annotation method for single-cell RNA-seq data,
together with three comparison methods: ARPL, OpenMax and PROSER.
CLaGNet learns a latent space with a residual MLP trained with cross-entropy plus a supervised
contrastive loss, fits a class-specific Gaussian mixture model (number of components chosen by BIC)
in that space, and rejects a cell as unknown when its class-conditional log-likelihood falls below
a class-wise threshold calibrated by ROC analysis on known-class validation data
