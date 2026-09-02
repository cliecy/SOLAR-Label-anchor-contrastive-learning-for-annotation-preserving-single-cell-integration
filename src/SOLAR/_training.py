"""Private epoch loop for SOLARModel.train."""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


def assemble_contrastive_features(
    z_x: torch.Tensor,
    z_t: Optional[torch.Tensor],
    labels: Sequence[str],
    text_anchors: Sequence[str],
    anchor_dedup: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    """Build SupCon features/labels, optionally keeping one anchor per class.

    Default (anchor_dedup=False / rep): stack per-cell expression and text
    embeddings as two views [N, 2, D] — each class anchor is repeated once per
    cell of that class in the contrastive pool.

    With anchor_dedup=True / uniq: keep all expression embeddings and append
    one embedding per unique text anchor, yielding a single-view pool of size
    N + K rather than 2N.
    """
    if z_t is None:
        return z_x.unsqueeze(1), list(labels)

    if not anchor_dedup:
        return torch.stack([z_x, z_t], dim=1), list(labels)

    unique_texts = list(dict.fromkeys(text_anchors))
    first_index: dict[str, int] = {}
    for index, text in enumerate(text_anchors):
        if text not in first_index:
            first_index[text] = index
    z_t_unique = torch.stack(
        [z_t[first_index[text]] for text in unique_texts], dim=0
    )
    unique_labels = [labels[first_index[text]] for text in unique_texts]
    pooled = torch.cat([z_x, z_t_unique], dim=0)
    pooled_labels = list(labels) + unique_labels
    return pooled.unsqueeze(1), pooled_labels


def run_epochs(
    model: torch.nn.Module,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    history: dict[str, list[float]],
    max_epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    gradient_clip_norm: Optional[float],
    anchor_dedup: bool = False,
) -> None:
    best_val_loss = float("inf")
    best_model_state = None
    wait_counter = 0

    for _ in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for expressions, text_anchors, labels in train_loader:
            expressions = expressions.to(device, non_blocking=True)
            z_x, z_t = model(expressions, text_anchors, device)
            features, feature_labels = assemble_contrastive_features(
                z_x, z_t, labels, text_anchors, anchor_dedup=anchor_dedup
            )
            loss = criterion(features, labels=feature_labels)

            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history["train_loss"].append(avg_train_loss)

        if val_loader is None:
            continue

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for expressions, text_anchors, labels in val_loader:
                expressions = expressions.to(device, non_blocking=True)
                z_x, z_t = model(expressions, text_anchors, device)
                features, feature_labels = assemble_contrastive_features(
                    z_x, z_t, labels, text_anchors, anchor_dedup=anchor_dedup
                )
                val_loss = criterion(features, labels=feature_labels)
                val_losses.append(float(val_loss.item()))

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history["val_loss"].append(avg_val_loss)

        if not np.isfinite(avg_val_loss):
            break

        if avg_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = avg_val_loss
            wait_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            wait_counter += 1
            if wait_counter >= early_stopping_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

