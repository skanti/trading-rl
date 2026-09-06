"""Plot the exact completed asset/reference sequence consumed by training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


from ml.dataset import MarketDayDataset


EXTENDED_SESSION_MINUTES = 16 * 60
DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEFAULT_BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def completed_sample(
    days: pd.DataFrame,
    data_dir: Path,
    sample_id: str,
    target_date: pd.Timestamp,
    context_days: int,
    rollout_size: int,
) -> dict:
    """Load one sample through the production forward-fill implementation."""
    symbol_days = days[days.sample_id.eq(sample_id)].copy()
    target = symbol_days[symbol_days.date.eq(target_date)].copy()
    if len(target) != 1:
        raise ValueError(
            f"expected one {sample_id} row for {target_date.date()}, got {len(target)}"
        )
    dataset = MarketDayDataset(
        days=target,
        calendar_days=symbol_days,
        data_dir=str(data_dir),
        window_size=context_days * EXTENDED_SESSION_MINUTES + 1,
        rollout_size=rollout_size,
        require_full_session=True,
    )
    return dataset[0]


def exact_observation_mask(data_dir: Path, sample_id: str, secs: np.ndarray) -> np.ndarray:
    """Return true where the source has a bar rather than a completed minute."""
    raw = np.load(data_dir / f"{sample_id}.npy", mmap_mode="r")
    raw_secs = np.asarray(raw[:, 0], dtype=np.int64)
    locations = np.searchsorted(raw_secs, secs)
    exact = locations < raw_secs.size
    exact[exact] &= raw_secs[locations[exact]] == secs[exact]
    return exact


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def normalized(prices: np.ndarray, anchor: int = 0) -> np.ndarray:
    return np.asarray(prices, dtype=np.float64) / float(prices[anchor]) * 100.0


def padded_bounds(*series: np.ndarray) -> tuple[float, float]:
    low = min(float(values.min()) for values in series)
    high = max(float(values.max()) for values in series)
    padding = max((high - low) * 0.08, 0.1)
    return low - padding, high + padding


def render_plot(
    asset_id: str,
    reference_id: str,
    target_date: pd.Timestamp,
    prices: np.ndarray,
    reference_prices: np.ndarray,
    secs: np.ndarray,
    asset_exact: np.ndarray,
    reference_exact: np.ndarray,
    context_days: int,
    rollout_size: int,
    anno: str,
    output: Path,
) -> None:
    context_ticks = context_days * EXTENDED_SESSION_MINUTES
    expected = context_ticks + rollout_size + 1
    for name, values in (
        ("asset prices", prices),
        ("reference prices", reference_prices),
        ("timestamps", secs),
    ):
        if values.shape != (expected,):
            raise ValueError(f"{name} must have shape {(expected,)}, got {values.shape}")

    origin = pd.Timestamp(anno, tz="UTC")
    times = (origin + pd.to_timedelta(secs, unit="s")).tz_convert("US/Eastern")
    context_dates = [
        times[index].date()
        for index in range(0, context_ticks, EXTENDED_SESSION_MINUTES)
    ]
    asset_name = asset_id.removeprefix("ST-")
    reference_name = reference_id.removeprefix("ST-")
    full_asset = normalized(prices)
    full_reference = normalized(reference_prices)
    day_asset = normalized(prices[context_ticks:])
    day_reference = normalized(reference_prices[context_ticks:])
    asset_imputed = ~asset_exact
    reference_imputed = ~reference_exact

    width, height = 2400, 1400
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title_font = load_font(DEFAULT_BOLD_FONT, 40)
    subtitle_font = load_font(DEFAULT_FONT, 23)
    axis_font = load_font(DEFAULT_FONT, 21)
    axis_bold_font = load_font(DEFAULT_BOLD_FONT, 22)
    small_font = load_font(DEFAULT_FONT, 18)
    legend_font = load_font(DEFAULT_FONT, 20)
    colors = {
        "asset": "#2563eb",
        "reference": "#16a34a",
        "asset_imputed": "#f59e0b",
        "reference_imputed": "#9333ea",
        "current": "#dc2626",
        "grid": "#d8dee9",
        "axis": "#475569",
        "text": "#0f172a",
        "muted": "#475569",
        "context_shade": "#eff6ff",
        "current_shade": "#fef2f2",
    }
    left, right = 155, width - 70
    full_top, full_bottom = 175, 820
    day_top, day_bottom = 1010, 1285

    def xcoord(index: int, count: int) -> float:
        return left + (right - left) * float(index) / float(count - 1)

    def ycoord(value: float, low: float, high: float, top: int, bottom: int) -> float:
        return bottom - (bottom - top) * (float(value) - low) / (high - low)

    def draw_axes(low: float, high: float, top: int, bottom: int) -> None:
        for tick in np.linspace(low, high, 6):
            y = ycoord(tick, low, high, top, bottom)
            draw.line((left, y, right, y), fill=colors["grid"], width=1)
            label = f"{tick:.2f}"
            box = draw.textbbox((0, 0), label, font=axis_font)
            draw.text(
                (left - 15 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
                label,
                fill=colors["muted"],
                font=axis_font,
            )
        draw.line((left, top, left, bottom), fill=colors["axis"], width=2)
        draw.line((left, bottom, right, bottom), fill=colors["axis"], width=2)

    def draw_series(
        values: np.ndarray,
        low: float,
        high: float,
        top: int,
        bottom: int,
        color: str,
        width_px: int,
        start: int = 0,
        total_count: int | None = None,
    ) -> None:
        count = int(total_count or len(values))
        points = [
            (xcoord(start + offset, count), ycoord(value, low, high, top, bottom))
            for offset, value in enumerate(values)
        ]
        draw.line(points, fill=color, width=width_px, joint="curve")

    draw.text(
        (left, 42),
        f"{asset_name} with {reference_name} reference: completed training sequence",
        fill=colors["text"],
        font=title_font,
    )
    draw.text(
        (left, 96),
        f"{context_days} completed extended sessions (04:00–19:59 ET) + current regular session on {target_date.date()}",
        fill=colors["muted"],
        font=subtitle_font,
    )

    full_low, full_high = padded_bounds(full_asset, full_reference)
    for day_index in range(context_days):
        x0 = xcoord(day_index * EXTENDED_SESSION_MINUTES, expected)
        x1 = xcoord(min((day_index + 1) * EXTENDED_SESSION_MINUTES, expected - 1), expected)
        if day_index % 2 == 0:
            draw.rectangle((x0, full_top, x1, full_bottom), fill=colors["context_shade"])
        draw.line((x0, full_top, x0, full_bottom), fill="#94a3b8", width=1)
    trading_x = xcoord(context_ticks, expected)
    draw.rectangle((trading_x, full_top, right, full_bottom), fill=colors["current_shade"])
    draw_axes(full_low, full_high, full_top, full_bottom)
    draw_series(full_asset, full_low, full_high, full_top, full_bottom, colors["asset"], 3)
    draw_series(full_reference, full_low, full_high, full_top, full_bottom, colors["reference"], 3)
    draw.line((trading_x, full_top, trading_x, full_bottom), fill=colors["current"], width=4)

    for mask, values, color in (
        (asset_imputed, full_asset, colors["asset_imputed"]),
        (reference_imputed, full_reference, colors["reference_imputed"]),
    ):
        for index in np.flatnonzero(mask):
            x = xcoord(int(index), expected)
            y = ycoord(values[index], full_low, full_high, full_top, full_bottom)
            draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=color)

    for day_index, date in enumerate(context_dates):
        center = day_index * EXTENDED_SESSION_MINUTES + EXTENDED_SESSION_MINUTES // 2
        label = pd.Timestamp(date).strftime("%b %d")
        x = xcoord(center, expected)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, full_bottom + 16),
            label,
            fill=colors["muted"],
            font=small_font,
        )
    current_label = target_date.strftime("%b %d RTH")
    x = xcoord(context_ticks + rollout_size // 2, expected)
    box = draw.textbbox((0, 0), current_label, font=small_font)
    draw.text(
        (x - (box[2] - box[0]) / 2, full_bottom + 16),
        current_label,
        fill=colors["current"],
        font=small_font,
    )

    draw.text(
        (left, full_top - 39),
        "Full sequence, independently normalized to 100 at the first context minute",
        fill=colors["text"],
        font=axis_bold_font,
    )
    legend_x, legend_y = right - 680, full_top + 20
    for color, label in (
        (colors["asset"], asset_name),
        (colors["reference"], reference_name),
        (colors["asset_imputed"], f"{asset_name} forward-filled"),
        (colors["reference_imputed"], f"{reference_name} forward-filled"),
    ):
        draw.line((legend_x, legend_y + 10, legend_x + 42, legend_y + 10), fill=color, width=5)
        draw.text((legend_x + 55, legend_y - 2), label, fill=colors["text"], font=legend_font)
        legend_y += 35

    metadata = (
        f"{context_ticks:,} context minutes + {rollout_size + 1} current-day prices = {expected:,} observations"
        f"  |  forward-filled: {asset_name}={int(asset_imputed.sum())}, {reference_name}={int(reference_imputed.sum())}"
    )
    draw.text((left, full_bottom + 55), metadata, fill=colors["muted"], font=small_font)

    day_low, day_high = padded_bounds(day_asset, day_reference)
    draw_axes(day_low, day_high, day_top, day_bottom)
    draw_series(day_asset, day_low, day_high, day_top, day_bottom, colors["asset"], 4)
    draw_series(day_reference, day_low, day_high, day_top, day_bottom, colors["reference"], 4)
    for mask, values, color in (
        (asset_imputed[context_ticks:], day_asset, colors["asset_imputed"]),
        (reference_imputed[context_ticks:], day_reference, colors["reference_imputed"]),
    ):
        for index in np.flatnonzero(mask):
            x = xcoord(int(index), rollout_size + 1)
            y = ycoord(values[index], day_low, day_high, day_top, day_bottom)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    minute_ticks = [(0, "09:30")]
    minute_ticks.extend((minute, f"{hour:02d}:00") for hour, minute in zip(range(10, 17), range(30, 391, 60)))
    for offset, label in minute_ticks:
        x = xcoord(offset, rollout_size + 1)
        draw.line((x, day_top, x, day_bottom), fill=colors["grid"], width=1)
        box = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (x - (box[2] - box[0]) / 2, day_bottom + 14),
            label,
            fill=colors["muted"],
            font=axis_font,
        )
    draw.text(
        (left, day_top - 42),
        "Current day, independently normalized to 100 at 09:30",
        fill=colors["text"],
        font=axis_bold_font,
    )
    draw.text((left, day_bottom + 58), "US/Eastern time", fill=colors["muted"], font=axis_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a completed context + trading-day asset/reference sequence."
    )
    parser.add_argument("--sample_id", default="NVDA")
    parser.add_argument("--reference_sample_id", default="SPY")
    parser.add_argument("--date", required=True, help="Trading date in YYYY-MM-DD format")
    parser.add_argument("--days_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--context_days", type=int, default=10)
    parser.add_argument("--rollout_size", type=int, default=390)
    parser.add_argument("--anno", default="2010-01-01")
    args = parser.parse_args()
    if args.context_days < 1 or args.rollout_size < 1:
        parser.error("context_days and rollout_size must be positive")
    return args


def main() -> None:
    args = parse_args()
    target_date = pd.Timestamp(args.date)
    days_path = Path(args.days_path)
    data_dir = Path(args.data_dir)
    days = pd.read_csv(days_path)
    days["date"] = pd.to_datetime(days["date"], format="%Y-%m-%d")
    asset = completed_sample(
        days,
        data_dir,
        args.sample_id,
        target_date,
        args.context_days,
        args.rollout_size,
    )
    reference = completed_sample(
        days,
        data_dir,
        args.reference_sample_id,
        target_date,
        args.context_days,
        args.rollout_size,
    )
    asset_secs = np.asarray(asset["secs"], dtype=np.int64)
    reference_secs = np.asarray(reference["secs"], dtype=np.int64)
    if not np.array_equal(asset_secs, reference_secs):
        raise ValueError("asset and reference completion produced different timestamps")

    output = (
        Path(args.out_path)
        if args.out_path
        else Path("/tmp/trading")
        / f"{args.sample_id.lower()}_{args.reference_sample_id.lower()}_completed_{args.context_days}d_plus_{args.date}.png"
    )
    asset_exact = exact_observation_mask(data_dir, args.sample_id, asset_secs)
    reference_exact = exact_observation_mask(
        data_dir, args.reference_sample_id, reference_secs
    )
    render_plot(
        args.sample_id,
        args.reference_sample_id,
        target_date,
        np.asarray(asset["prices"], dtype=np.float64),
        np.asarray(reference["prices"], dtype=np.float64),
        asset_secs,
        asset_exact,
        reference_exact,
        args.context_days,
        args.rollout_size,
        args.anno,
        output,
    )
    context_ticks = args.context_days * EXTENDED_SESSION_MINUTES
    print(output)
    print(
        f"observations={asset_secs.size}, "
        f"asset_forward_filled={int((~asset_exact).sum())}, "
        f"reference_forward_filled={int((~reference_exact).sum())}, "
        f"asset_current_forward_filled={int((~asset_exact[context_ticks:]).sum())}, "
        f"reference_current_forward_filled={int((~reference_exact[context_ticks:]).sum())}"
    )


if __name__ == "__main__":
    main()
