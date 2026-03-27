"""
Irrigation Detection Pipeline
==============================

Takes hourly meter readings (meter_id, timestamp, reading), runs them through
the DilatedUNet1D model in 336-hour (14-day) windows, and outputs per-hour
irrigation predictions.

The model directly predicts irrigation volume from total consumption —
no rolling median or historical baseline needed.

Input:  DataFrame with columns (meter_id, timestamp, reading)
Output: DataFrame with columns (meter_id, timestamp, total_reading,
        irrigation_reading, non_irrigation_reading)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .models import DilatedUNet1D, WINDOW_SIZE

logger = logging.getLogger(__name__)

DEFAULT_GPU_BATCH = 2000  # Windows per GPU batch (avoid OOM)


def _predict_vectorized(
    model: DilatedUNet1D,
    consumption: np.ndarray,
    orig_lengths: list[int],
    device: str = "cpu",
    gpu_batch: int = DEFAULT_GPU_BATCH,
) -> np.ndarray:
    """
    Mega-batch prediction: pad all meter arrays to 336-hour blocks,
    concat into one tensor, predict in one pass, slice back.

    Args:
        model: Loaded DilatedUNet1D.
        consumption: Flat concatenated array of padded consumption values.
        orig_lengths: Original (unpadded) length per meter.
        device: Torch device string.
        gpu_batch: Max windows per forward pass.

    Returns:
        Flat array of irrigation predictions, trimmed to original lengths.
    """
    num_total_windows = len(consumption) // WINDOW_SIZE
    input_tensor = (
        torch.FloatTensor(consumption)
        .reshape(num_total_windows, 1, WINDOW_SIZE)
        .to(device)
    )

    preds_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, num_total_windows, gpu_batch):
            batch = input_tensor[i : i + gpu_batch]
            p = model(batch)
            preds_list.append(p.cpu().numpy().flatten())

    flat_preds = np.concatenate(preds_list)

    # Unpad: slice back to original meter lengths
    final_results = []
    cursor = 0
    for length in orig_lengths:
        rem = length % WINDOW_SIZE
        padded_len = length if rem == 0 else length + (WINDOW_SIZE - rem)
        final_results.append(flat_preds[cursor : cursor + length])
        cursor += padded_len

    return np.concatenate(final_results)


def detect_irrigation(
    df: pd.DataFrame,
    model: DilatedUNet1D,
    device: str = "cpu",
    source_label: Optional[str] = None,
    gpu_batch: int = DEFAULT_GPU_BATCH,
) -> pd.DataFrame:
    """
    Run irrigation detection on hourly meter readings.

    Args:
        df: Input DataFrame. Required columns:
            - meter_id (str):   Unique meter identifier
            - timestamp (str/datetime): Hourly reading timestamp
            - reading (float):  Consumption value for that hour (non-negative)
        model: Pre-trained DilatedUNet1D instance.
        device: Torch device string ('cpu' or 'cuda').
        source_label: Optional label added as a 'source' column in the output.
        gpu_batch: Max inference windows per forward pass.

    Returns:
        DataFrame with columns:
            meter_id, timestamp, total_reading,
            irrigation_reading, non_irrigation_reading
            (and optionally 'source' if source_label is provided)
    """
    logger.info("=" * 60)
    logger.info("IRRIGATION DETECTION PIPELINE START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Validate and prepare input
    # ------------------------------------------------------------------
    logger.info("[1/4] Preparing hourly consumption data...")

    required_cols = {"meter_id", "timestamp", "reading"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame missing required columns: {missing}. "
            f"Expected columns: meter_id, timestamp, reading"
        )

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed", utc=True)
    data["meter_id"] = data["meter_id"].astype(str)
    data = data.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Clean readings: fill NaN with 0, clamp negatives
    data["consumption"] = data["reading"].fillna(0).clip(lower=0).astype(np.float32)

    n_accounts = data["meter_id"].nunique()
    logger.info(f"  {len(data):,} hourly rows, {n_accounts} meters")

    # ------------------------------------------------------------------
    # 2. Build padded arrays per meter
    # ------------------------------------------------------------------
    logger.info("[2/4] Building padded arrays per meter...")
    grouped = data.groupby("meter_id", sort=True)

    arrays = []
    orig_lengths = []

    for _, group in grouped:
        v = group["consumption"].values.astype(np.float32)
        length = len(v)

        # Pad to multiple of WINDOW_SIZE
        rem = length % WINDOW_SIZE
        if rem > 0:
            pad_len = WINDOW_SIZE - rem
            v = np.pad(v, (0, pad_len), "edge")

        arrays.append(v)
        orig_lengths.append(length)

    if not arrays:
        logger.warning("No data to process")
        return pd.DataFrame()

    flat_input = np.concatenate(arrays)
    n_windows = len(flat_input) // WINDOW_SIZE
    logger.info(f"  {len(arrays)} meters -> {n_windows} windows ({len(flat_input):,} hours)")

    # ------------------------------------------------------------------
    # 3. Run model inference
    # ------------------------------------------------------------------
    logger.info("[3/4] Running DilatedUNet1D inference...")
    irrigation_preds = _predict_vectorized(
        model, flat_input, orig_lengths, device, gpu_batch
    )

    # Physics clamp: irrigation >= 0 and <= total consumption
    irrigation_preds = irrigation_preds.clip(min=0)
    total_consumption = data["consumption"].values
    irrigation_preds = np.minimum(irrigation_preds, total_consumption)

    data["irrigation_reading"] = irrigation_preds.astype(np.float32)
    data["non_irrigation_reading"] = (
        (data["consumption"] - data["irrigation_reading"]).clip(lower=0).astype(np.float32)
    )

    # ------------------------------------------------------------------
    # 4. Format output
    # ------------------------------------------------------------------
    logger.info("[4/4] Formatting output...")
    output_cols = {
        "meter_id": data["meter_id"],
        "timestamp": data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "total_reading": data["consumption"].round(4),
        "irrigation_reading": data["irrigation_reading"].round(4),
        "non_irrigation_reading": data["non_irrigation_reading"].round(4),
    }

    if source_label:
        output_cols["source"] = source_label

    output = pd.DataFrame(output_cols)
    output = output.dropna(subset=["timestamp", "meter_id"])
    output = output.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)

    # Summary stats
    n_irrig = (output["irrigation_reading"] > 0).sum()
    total_irrig = output["irrigation_reading"].sum()
    total_cons = output["total_reading"].sum()
    pct = 100 * total_irrig / max(total_cons, 1)
    logger.info(f"  Output: {len(output):,} rows, {output['meter_id'].nunique()} meters")
    logger.info(f"  Irrigation hours: {n_irrig:,} ({100 * n_irrig / max(len(output), 1):.2f}%)")
    logger.info(f"  Irrigation volume: {total_irrig:,.0f} / {total_cons:,.0f} ({pct:.1f}%)")
    logger.info("PIPELINE COMPLETE")

    return output
