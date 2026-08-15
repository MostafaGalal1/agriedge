"""Deep baselines matching the architectures the Edge-IIoTset literature reports.

The headline results on this dataset come from deep models - CNNs, DNNs and
hybrids - rather than from the classical estimators of ``zoo.py``. Showing that
the serialisation artifact inflates classical models is therefore only half the
argument; a reviewer is entitled to ask whether a deep model would have been
robust to it. It is not: a deep model consuming the leaked one-hot features
reads them exactly as a decision tree does.

Two architectures are provided. The MLP is the standard tabular baseline. The
1D-CNN treats the feature vector as a sequence, which is the construction used
by several published Edge-IIoTset papers and which we include so the comparison
is against what the literature actually ran, not a strawman.

PyTorch is an optional dependency; absence raises a clear instruction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

try:  # pragma: no cover - environment-dependent
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the deep baselines. "
            "Install it with: pip install 'agriedge[torch]'"
        )


@dataclass(frozen=True)
class DeepConfig:
    """Training schedule for the deep baselines.

    Attributes:
        epochs: Passes over the training set.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        hidden_sizes: MLP hidden widths.
        conv_channels: 1D-CNN channel widths.
        kernel_size: 1D-CNN kernel width.
        dropout: Dropout probability applied after each hidden block.
        seed: Seed for initialisation and shuffling.
    """

    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (256, 128, 64)
    conv_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 3
    dropout: float = 0.2
    seed: int = 20260814

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be at least 1.")


def build_mlp(n_features: int, config: DeepConfig):
    """Feed-forward baseline for tabular input."""
    _require_torch()
    if n_features <= 0:
        raise ValueError(f"n_features must be positive; got {n_features}.")

    layers: list[nn.Module] = []
    in_dim = n_features
    for width in config.hidden_sizes:
        layers.extend(
            [nn.Linear(in_dim, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(config.dropout)]
        )
        in_dim = width
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


class Conv1DClassifier(nn.Module if TORCH_AVAILABLE else object):
    """1D-CNN over the feature vector, as used in several published baselines."""

    def __init__(self, n_features: int, config: DeepConfig):
        _require_torch()
        super().__init__()
        if n_features < config.kernel_size:
            raise ValueError(
                f"n_features ({n_features}) must be at least kernel_size "
                f"({config.kernel_size})."
            )

        blocks: list[nn.Module] = []
        in_channels = 1
        for channels in config.conv_channels:
            blocks.extend(
                [
                    nn.Conv1d(in_channels, channels, config.kernel_size, padding="same"),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_channels = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(config.dropout), nn.Linear(in_channels, 2)
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x.unsqueeze(1))))


@dataclass(frozen=True)
class DeepResult:
    """Scores and cost for one trained deep baseline."""

    model: str
    protocol: str
    n_parameters: int
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    train_seconds: float
    device: str

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "n_parameters": self.n_parameters,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "train_seconds": self.train_seconds,
            "device": self.device,
        }


def train_and_score(
    architecture: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: DeepConfig | None = None,
    *,
    protocol: str,
    verbose: bool = True,
) -> DeepResult:
    """Train one deep baseline and score it on the held-out set.

    Args:
        architecture: ``"mlp"`` or ``"cnn1d"``.
        x_train: Scaled training features.
        y_train: Binary training labels.
        x_test: Scaled test features.
        y_test: Binary test labels.
        config: Training schedule.
        protocol: Recorded on the result row.
        verbose: Print per-epoch loss.

    Raises:
        RuntimeError: if PyTorch is unavailable.
        ValueError: on unknown architecture or malformed input.
    """
    _require_torch()
    settings = config or DeepConfig()

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Feature and label lengths disagree.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set contains a single class.")

    torch.manual_seed(settings.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    n_features = x_train.shape[1]
    if architecture == "mlp":
        model = build_mlp(n_features, settings)
    elif architecture == "cnn1d":
        model = Conv1DClassifier(n_features, settings)
    else:
        raise ValueError(
            f"Unknown architecture {architecture!r}; expected 'mlp' or 'cnn1d'."
        )
    model = model.to(device)

    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        ),
        batch_size=settings.batch_size,
        shuffle=True,
        drop_last=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    start = time.perf_counter()
    model.train()
    for epoch in range(settings.epochs):
        losses: list[float] = []
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
        if verbose and (epoch % 5 == 0 or epoch == settings.epochs - 1):
            print(f"    epoch {epoch:02d} loss={np.mean(losses):.5f}")
    train_seconds = time.perf_counter() - start

    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, len(x_test), settings.batch_size):
            batch = torch.as_tensor(
                x_test[start_index : start_index + settings.batch_size],
                dtype=torch.float32,
            ).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(predictions)

    return DeepResult(
        model=architecture,
        protocol=protocol,
        n_parameters=int(sum(p.numel() for p in model.parameters())),
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        accuracy=float((y_pred == np.asarray(y_test)).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        train_seconds=train_seconds,
        device=str(device),
    )


def to_frame(results: list[DeepResult] | tuple[DeepResult, ...]) -> pd.DataFrame:
    """Tabulate deep-baseline results."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.as_row() for r in results])
