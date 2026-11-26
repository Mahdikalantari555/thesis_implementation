"""Self-supervised models: GRU-AE, Masked Transformer, STS-Net, Tiny-CNN-Mamba, Time2Agri, JEPA."""

from __future__ import annotations

from .cnn_mamba import BidirectionalMambaBlock, SelectiveSSM, TinyCNNMamba
from .gru_ae import GRUAutoencoder
from .jepa import JEPAModel
from .masked_tf import MaskedTransformer
from .pretexts import (
    PRETEXT_ARMS,
    PretextTransformer,
    Time2AgriModel,
    info_nce_loss,
    make_time2agri_views,
    score_cycle_pretext,
    train_pretext_arm,
)
from .sts_net import STSNetAutoencoder
from .time2agri_archs import (
    ARMS as TIME2AGRI_ARMS,
    Time2AgriFF,
    Time2AgriFP,
    Time2AgriTD,
    build_fp_targets,
    build_time2agri_arm,
    fft_ndvi_target,
)
from .tokenizer import CycleTokenizer, tokenize_cycle

__all__ = [
    "PRETEXT_ARMS",
    "TIME2AGRI_ARMS",
    "BidirectionalMambaBlock",
    "Time2AgriFF",
    "Time2AgriFP",
    "Time2AgriTD",
    "build_fp_targets",
    "build_time2agri_arm",
    "fft_ndvi_target",
    "CycleTokenizer",
    "GRUAutoencoder",
    "JEPAModel",
    "MaskedTransformer",
    "PretextTransformer",
    "STSNetAutoencoder",
    "SelectiveSSM",
    "Time2AgriModel",
    "TinyCNNMamba",
    "info_nce_loss",
    "make_time2agri_views",
    "score_cycle_pretext",
    "tokenize_cycle",
    "train_pretext_arm",
]
