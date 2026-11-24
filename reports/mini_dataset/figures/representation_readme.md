# How to read these figures

- `representation_structure.png`: each panel uses its two most discriminative dimensions (Fisher ratio for the colour variable) of the frozen SSL / raw / random-init embeddings. Top row coloured by farm, bottom row by stage. Tighter clusters = the representation carries that structure better than the others.
- `representation_cross_year.png`: left, one farm/stage with train (blue) vs test (red) years overlaid on the SSL top-2 dims; right, the two metric families side by side.

**Renderer note:** UMAP and t-SNE crash natively in this conda environment, so a deterministic pure-Python 2-D projection (top-2 Fisher-ratio dimensions) is used instead. Do not label it UMAP/t-SNE.

**Do not claim:** stress detection, early warning, geographic generalization, or SSL superiority. L1/L2 screening on 3 adjacent farms with weak suspicion labels; CWSI shares inputs with the model. The retrieval metric has a failed negative control and the embedding has low effective rank.
