"""Stage-2 pixel scoring via ONNXRuntime CPU (change tasks 2.2 + 3.3).

Exports the trained pixel CNN-AE to ONNX once, then scores the top-K
suspicious (farm, date) canvases with the ONNX Runtime CPU execution
provider. Torch/ORT numeric parity is asserted by
``tests/unit/test_probe.py::test_cnn_ae_onnx_parity`` (atol 1e-3, fp32).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .cnn_ae import PixelCNNAE, make_input


def export_onnx(model: PixelCNNAE, out_path: Path, canvas: int = 48,
                opset: int = 17) -> Path:
    """Export (static (1, 8, S, S) graph)."""
    model.eval()
    dummy = torch.zeros(1, 8, canvas, canvas)
    torch.onnx.export(
        model, dummy, str(out_path), opset_version=opset, dynamo=False,
        input_names=["canvas"], output_names=["recon"],
        dynamic_axes=None,
    )
    return out_path


def ort_session(onnx_path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts,
                                providers=["CPUExecutionProvider"])
    return sess


def ort_score_canvas(sess, canvas: np.ndarray, valid: np.ndarray,
                     in_field: np.ndarray, norm: dict,
                     band_sigma: np.ndarray) -> tuple[float, np.ndarray]:
    """One (farm, date) → (field-date score, pixel score map (S, S), NaN outside).

    Compared in **standardized units** (identical to the Stage-1 torch scorer:
    target = standardized canvas, residual divided by σ of standardized train
    residuals) so Stage-1 and Stage-2 scores are the same quantity and
    torch-vs-ORT parity is directly checkable.
    """
    x, loss_mask = make_input(canvas, valid, in_field, norm)
    if not loss_mask.any():
        return float("nan"), np.full(canvas.shape[1:], np.nan, dtype=np.float32)
    recon = sess.run(["recon"], {"canvas": x[None].astype(np.float32)})[0][0]
    target = x[:7]  # standardized canvas (NaN→0 placeholder is masked out)
    diff = np.abs(recon - target) / band_sigma[:, None, None]
    per_px = diff.mean(axis=0)                       # (S, S) mean over bands
    per_px[~loss_mask] = np.nan
    with np.errstate(all="ignore"):
        fd = float(np.nanmean(per_px))
    return fd, per_px.astype(np.float32)
