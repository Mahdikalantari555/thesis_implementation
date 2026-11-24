# farm-subset-anomaly-probe

Small-scale (32-farm subset, CPU-only) anomaly-scoring probe: pixel-wise CNN-AE + Barlow Twins trajectories + GRU-AE baseline + B2 no-NN baseline; produces anomaly scores. Combines with date-centric change optionally; feasible on IRIS-XE-less laptop via small models (<=250k params, <=15 min/seed) and ONNXRuntime CPU inference.
