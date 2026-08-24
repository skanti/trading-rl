"""Binary price-direction features, labels, model, and metrics.

Supported experiments either expose one stock/reference-relative channel or
two independently normalized stock and reference channels. In both layouts the
labelled future tick is excluded from the observation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import model


RELATIVE_FEATURE_NAMES = ("relative_log_price",)
DUAL_PRICE_FEATURE_NAMES = ("stock_log_price", "reference_log_price")
# Backward-compatible names for the original relative experiment and tests.
CLASSIFY_FEATURE_NAMES = RELATIVE_FEATURE_NAMES
CLASSIFY_FEATURE_DIM = len(CLASSIFY_FEATURE_NAMES)
WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday")
CLASSIFY_SCALAR_NAMES = (
    "anchor_progress",
    *(f"weekday_{name}" for name in WEEKDAY_NAMES),
)
CLASSIFY_SCALAR_DIM = len(CLASSIFY_SCALAR_NAMES)


def classify_feature_names(feature_mode: str) -> tuple[str, ...]:
    mode = str(feature_mode)
    if mode == "relative":
        return RELATIVE_FEATURE_NAMES
    if mode == "dual_normalized":
        return DUAL_PRICE_FEATURE_NAMES
    raise ValueError("feature_mode must be 'relative' or 'dual_normalized'")


def relative_log_prices(prices: torch.Tensor, reference_prices: torch.Tensor) -> torch.Tensor:
    """``log(stock) - log(reference)``, the only market quantity exposed.

    Absolute price, the reference's own path, and symbol identity all cancel or
    are excluded, so a model cannot recognize the date from the reference's
    signature and memorize the answer.
    """
    if prices.shape != reference_prices.shape or prices.ndim != 2:
        raise ValueError("prices and reference_prices must share shape (batch, sequence)")
    if not torch.isfinite(prices).all() or not torch.isfinite(reference_prices).all():
        raise ValueError("prices must be finite")
    if (prices <= 0).any() or (reference_prices <= 0).any():
        raise ValueError("prices must be positive")
    return torch.log(prices.float()) - torch.log(reference_prices.float())


def build_relative_features(
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    """One relative-price channel of shape ``(batch, window, 1)``.

    The final column of ``prices`` is the labelled tick and is excluded here, so
    the observation stops at the anchor and cannot see its own answer. Anchoring
    the log stock/reference ratio at zero makes each value a relative return to
    the chosen intraday time, while never exposing either price path alone.
    """
    relative = relative_log_prices(prices, reference_prices)[:, :-1]
    if relative.shape[1] < 2:
        raise ValueError("the window must hold at least two ticks")
    scale = float(price_feature_scale)
    anchored = (relative - relative[:, -1:]) * scale
    return anchored.unsqueeze(-1)


def build_dual_price_features(
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    """Independently normalized stock and reference channels.

    Both paths are log-normalized to their 15:55 entry value. The future target
    tick is excluded, and scaling either raw price series by a constant leaves
    the corresponding channel unchanged.
    """
    if prices.shape != reference_prices.shape or prices.ndim != 2:
        raise ValueError("prices and reference_prices must share shape (batch, sequence)")
    if not torch.isfinite(prices).all() or not torch.isfinite(reference_prices).all():
        raise ValueError("prices must be finite")
    if (prices <= 0).any() or (reference_prices <= 0).any():
        raise ValueError("prices must be positive")
    stock = torch.log(prices.float())[:, :-1]
    reference = torch.log(reference_prices.float())[:, :-1]
    if stock.shape[1] < 2:
        raise ValueError("the window must hold at least two ticks")
    scale = float(price_feature_scale)
    stock = (stock - stock[:, -1:]) * scale
    reference = (reference - reference[:, -1:]) * scale
    return torch.stack((stock, reference), dim=-1)


def build_classify_features(
    prices: torch.Tensor,
    reference_prices: torch.Tensor,
    feature_mode: str,
    price_feature_scale: float = 100.0,
) -> torch.Tensor:
    if str(feature_mode) == "relative":
        return build_relative_features(prices, reference_prices, price_feature_scale)
    if str(feature_mode) == "dual_normalized":
        return build_dual_price_features(prices, reference_prices, price_feature_scale)
    raise ValueError("feature_mode must be 'relative' or 'dual_normalized'")


def relative_labels(prices: torch.Tensor, reference_prices: torch.Tensor) -> torch.Tensor:
    """1.0 when the stock outperforms the reference over the horizon.

    Comparing log ratios is sign-identical to comparing the normalized prices
    directly: ``P_s(t+d)/P_s(t) > P_r(t+d)/P_r(t)`` iff the log ratio rose.
    """
    relative = relative_log_prices(prices, reference_prices)
    return (relative[:, -1] > relative[:, -2]).float()


def stock_direction_labels(prices: torch.Tensor) -> torch.Tensor:
    """1.0 when the stock target price is above its entry-anchor price."""
    if prices.ndim != 2 or prices.shape[1] < 2:
        raise ValueError("prices must have shape (batch, sequence>=2)")
    if not torch.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")
    return prices[:, -1].gt(prices[:, -2]).float()


def classify_labels(
    prices: torch.Tensor, reference_prices: torch.Tensor, target_mode: str
) -> torch.Tensor:
    if str(target_mode) == "relative_direction":
        return relative_labels(prices, reference_prices)
    if str(target_mode) == "stock_direction":
        return stock_direction_labels(prices)
    raise ValueError("target_mode must be 'relative_direction' or 'stock_direction'")


def build_classify_scalars(
    anchor_progress: torch.Tensor, weekday: torch.Tensor
) -> torch.Tensor:
    """Current-session clock plus an explicit Monday--Friday one-hot vector."""
    if anchor_progress.ndim != 1 or weekday.shape != anchor_progress.shape:
        raise ValueError("anchor_progress and weekday must share one-dimensional shape")
    if not torch.isfinite(anchor_progress).all():
        raise ValueError("anchor_progress must be finite")
    if anchor_progress.numel() and (
        anchor_progress.lt(0).any() or anchor_progress.gt(1).any()
    ):
        raise ValueError("anchor_progress must be in [0, 1]")
    weekday_long = weekday.to(torch.long)
    if weekday.numel() and (
        weekday.ne(weekday_long).any()
        or weekday_long.lt(0).any()
        or weekday_long.ge(len(WEEKDAY_NAMES)).any()
    ):
        raise ValueError("weekday must contain Monday=0 through Friday=4")
    weekday_one_hot = F.one_hot(
        weekday_long, num_classes=len(WEEKDAY_NAMES)
    ).to(anchor_progress.dtype)
    return torch.cat((anchor_progress.unsqueeze(-1), weekday_one_hot), dim=-1)


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Win rate and the references needed to read it.

    ``win_rate`` alone is uninterpretable: a constant predictor already scores
    ``majority_rate``. ``win_rate_confident`` is the accuracy on the most
    confident fifth, which is the number that matters if the signal is ever
    traded selectively.
    """
    if logits.shape != labels.shape or logits.ndim != 1:
        raise ValueError("logits and labels must be matching 1-D tensors")
    if not logits.numel():
        raise ValueError("logits must be non-empty")
    logits = logits.detach().float()
    labels = labels.detach().float()
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("logits and labels must be finite")
    if not labels.eq(0).logical_or(labels.eq(1)).all():
        raise ValueError("labels must be binary")

    predictions = logits.ge(0)
    positives = labels.bool()
    negatives = ~positives
    correct = predictions.eq(positives)
    base_rate = labels.mean()
    count = labels.numel()

    positive_count = int(positives.sum())
    negative_count = count - positive_count
    true_positive = int((predictions & positives).sum())
    true_negative = int((~predictions & negatives).sum())
    false_positive = int((predictions & negatives).sum())
    false_negative = int((~predictions & positives).sum())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    precision = true_positive / precision_denominator if precision_denominator else float("nan")
    recall = true_positive / recall_denominator if recall_denominator else float("nan")
    specificity = (
        true_negative / specificity_denominator if specificity_denominator else float("nan")
    )
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if recall_denominator and specificity_denominator
        else float("nan")
    )
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else float("nan")

    if positive_count and negative_count:
        # Mann-Whitney U with average ranks for ties. Assigning arbitrary ranks
        # to equal logits can otherwise make a constant classifier look useful.
        order = logits.argsort()
        sorted_logits = logits[order]
        _, tie_counts = torch.unique_consecutive(sorted_logits, return_counts=True)
        tie_ends = tie_counts.cumsum(0)
        tie_starts = tie_ends - tie_counts
        average_ranks = (tie_starts + tie_ends + 1).to(logits.dtype) / 2.0
        sorted_ranks = torch.repeat_interleave(average_ranks, tie_counts)
        ranks = torch.empty_like(sorted_ranks)
        ranks[order] = sorted_ranks
        rank_sum = ranks[positives].sum()
        auc = float(
            (rank_sum - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )
    else:
        auc = float("nan")

    confident = max(1, (count + 4) // 5)
    top = logits.abs().argsort(descending=True)[:confident]
    probabilities = logits.sigmoid()
    return {
        "win_rate": float(correct.float().mean()),
        "win_rate_confident": float(correct[top].float().mean()),
        "base_rate": float(base_rate),
        "majority_rate": float(max(base_rate, 1.0 - base_rate)),
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "brier_score": float((probabilities - labels).square().mean()),
        "predicted_positive_rate": float(predictions.float().mean()),
        "confident_fraction": confident / count,
        "samples": float(count),
    }


class RelativeDirectionClassifier(model.WindowMLP):
    """Flat-window MLP emitting one binary-direction logit per window."""

    def __init__(
        self,
        window_size: int,
        feature_dim: int = CLASSIFY_FEATURE_DIM,
        hidden_dim: int = 512,
        depth: int = 4,
        scalar_dim: int = CLASSIFY_SCALAR_DIM,
    ):
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be positive")
        super().__init__(window_size, feature_dim, hidden_dim, 1, depth, scalar_dim)
        # A small head starts the classifier near p=0.5 rather than at an
        # arbitrary confident guess.
        torch.nn.init.orthogonal_(self.main[-1].weight, gain=0.01)

    def forward(
        self, inputs: torch.Tensor, scalars: torch.Tensor | None = None
    ) -> torch.Tensor:
        flat, leading = self.forward_features(inputs, scalars)
        return self.main(flat).reshape(*leading)
