"""FedAvg simulation over the AgriEdge benchmark.

Implements McMahan et al.'s federated averaging with a compact MLP sized for
a farm gateway. Clients never exchange data; each round they receive the
global weights, train locally, and return weights that the server averages in
proportion to local sample count.

PyTorch is an optional dependency. Importing this module without it raises a
clear instruction rather than an ImportError from deep inside a call stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from agriedge.config import RANDOM_SEED
from agriedge.federated.partition import Partition

try:  # pragma: no cover - exercised by environment, not by tests
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the federated simulation. "
            "Install it with: pip install 'agriedge[torch]'"
        )


@dataclass(frozen=True)
class FedConfig:
    """Federated training schedule.

    Attributes:
        rounds: Number of communication rounds.
        local_epochs: Local passes per client per round.
        batch_size: Local mini-batch size.
        learning_rate: Local SGD learning rate.
        hidden_sizes: Widths of the MLP's hidden layers.
        client_fraction: Fraction of clients sampled each round.
        seed: Seed for initialisation and client sampling.
    """

    rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 0.01
    hidden_sizes: tuple[int, ...] = (64, 32)
    client_fraction: float = 1.0
    seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if self.rounds <= 0 or self.local_epochs <= 0:
            raise ValueError("rounds and local_epochs must be positive.")
        if not 0.0 < self.client_fraction <= 1.0:
            raise ValueError("client_fraction must lie in (0, 1].")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")


@dataclass(frozen=True)
class RoundRecord:
    """Global-model scores after one communication round."""

    round_index: int
    clients_sampled: int
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    mean_local_loss: float


def build_model(n_features: int, config: FedConfig):
    """Return a freshly initialised MLP for binary classification."""
    _require_torch()
    if n_features <= 0:
        raise ValueError(f"n_features must be positive; got {n_features}.")

    layers: list[nn.Module] = []
    in_dim = n_features
    for width in config.hidden_sizes:
        layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
        in_dim = width
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


def _state_to_vector(state: Mapping[str, "torch.Tensor"]) -> dict:
    return {k: v.detach().clone() for k, v in state.items()}


def _average_states(states: list[dict], weights: list[int]) -> dict:
    """Weighted average of client state dicts, by local sample count."""
    if not states:
        raise ValueError("No client states to average.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Client weights must sum to a positive value.")

    averaged = {k: torch.zeros_like(v) for k, v in states[0].items()}
    for state, weight in zip(states, weights):
        share = weight / total
        for key, tensor in state.items():
            averaged[key] += tensor * share
    return averaged


def _train_local(model, loader, config: FedConfig, device) -> float:
    """Run local epochs, returning mean loss over the final epoch."""
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    model.train()

    last_epoch_losses: list[float] = []
    for _ in range(config.local_epochs):
        last_epoch_losses = []
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimiser.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimiser.step()
            last_epoch_losses.append(float(loss.item()))
    return float(np.mean(last_epoch_losses)) if last_epoch_losses else float("nan")


def _evaluate(model, x_test, y_test, device) -> tuple[float, float, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x_test, dtype=torch.float32).to(device))
        predictions = logits.argmax(dim=1).cpu().numpy()
    return (
        float(accuracy_score(y_test, predictions)),
        float(balanced_accuracy_score(y_test, predictions)),
        float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    )


def run_fedavg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    partition: Partition,
    config: FedConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[RoundRecord, ...]:
    """Run FedAvg and return the global model's trajectory.

    Args:
        x_train: Training features, already scaled.
        y_train: Binary training labels.
        x_test: Held-out features for global evaluation.
        y_test: Held-out labels.
        partition: Client assignment over ``x_train`` rows.
        config: Training schedule.
        verbose: Print per-round progress.

    Raises:
        RuntimeError: if PyTorch is unavailable.
        ValueError: if shapes disagree or a client index is out of range.
    """
    _require_torch()
    settings = config or FedConfig()

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train differ in length.")
    for name, index in partition.assignments.items():
        if index.max(initial=-1) >= len(x_train):
            raise ValueError(f"Client {name!r} indexes beyond the training set.")

    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"fedavg on {device} | clients={partition.n_clients} "
              f"| scheme={partition.scheme}")

    global_model = build_model(x_train.shape[1], settings).to(device)
    loaders = _build_loaders(x_train, y_train, partition, settings)

    names = list(partition.client_names)
    n_sampled = max(1, int(round(settings.client_fraction * len(names))))
    history: list[RoundRecord] = []

    for round_index in range(settings.rounds):
        selected = rng.choice(names, size=n_sampled, replace=False)
        states, weights, losses = [], [], []

        for name in selected:
            local = build_model(x_train.shape[1], settings).to(device)
            local.load_state_dict(_state_to_vector(global_model.state_dict()))
            losses.append(_train_local(local, loaders[name], settings, device))
            states.append(_state_to_vector(local.state_dict()))
            weights.append(len(partition.assignments[name]))

        global_model.load_state_dict(_average_states(states, weights))
        accuracy, balanced, f1 = _evaluate(global_model, x_test, y_test, device)
        record = RoundRecord(
            round_index=round_index,
            clients_sampled=len(selected),
            accuracy=accuracy,
            balanced_accuracy=balanced,
            f1_macro=f1,
            mean_local_loss=float(np.mean(losses)),
        )
        history.append(record)
        if verbose:
            print(
                f"  round {round_index:02d}  acc={accuracy:.4f} "
                f"bal_acc={balanced:.4f} f1_macro={f1:.4f} "
                f"loss={record.mean_local_loss:.4f}"
            )

    return tuple(history)


def _build_loaders(
    x_train: np.ndarray, y_train: np.ndarray, partition: Partition, config: FedConfig
) -> dict[str, "DataLoader"]:
    """Materialise one DataLoader per client."""
    loaders: dict[str, DataLoader] = {}
    for name, index in partition.assignments.items():
        dataset = TensorDataset(
            torch.as_tensor(x_train[index], dtype=torch.float32),
            torch.as_tensor(y_train[index], dtype=torch.long),
        )
        loaders[name] = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
        )
    return loaders


def history_to_frame(history: tuple[RoundRecord, ...]) -> pd.DataFrame:
    """Tabulate a FedAvg trajectory."""
    if not history:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "round": r.round_index,
                "clients_sampled": r.clients_sampled,
                "accuracy": r.accuracy,
                "balanced_accuracy": r.balanced_accuracy,
                "f1_macro": r.f1_macro,
                "mean_local_loss": r.mean_local_loss,
            }
            for r in history
        ]
    )
