"""
FEDD dataset reproduction + validation package.

This script generates artificial time-series datasets following the FEDD paper:
Cavalcante, Minku & Oliveira (2016), Tables I and II.

It also optionally runs the uploaded FEDD detector implementation against each
generated series, when scikit-multiflow and statsmodels are available.

Outputs:
- data/*.csv: generated time series
- metadata.csv: dataset description and true drift points
- detection_results.csv: optional FEDD detections, if --run-fedd is used

CSV columns:
- t: time index, 1..12000
- value: generated observation
- concept: true concept id, 1..4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


N_SERIES_PER_SUBSET = 20
CONCEPT_LEN = 3000
TOTAL_LEN = 12000
GRADUAL_LEN = 300
DRIFT_POINTS = [3000, 6000, 9000]


LINEAR_GROUPS = {
    "linear1_ar4": [
        ([0.9, -0.2, 0.8, -0.5], 0.5),
        ([-0.3, 1.4, 0.4, -0.5], 1.5),
        ([1.5, -0.4, -0.3, 0.2], 2.5),
        ([-0.1, 1.4, 0.4, -0.7], 3.5),
    ],
    "linear2_ar6": [
        ([1.1, -0.6, 0.8, -0.5, -0.1, 0.3], 0.5),
        ([-0.1, 1.2, 0.4, 0.3, -0.2, -0.6], 1.5),
        ([1.2, -0.4, -0.3, 0.7, -0.6, 0.4], 2.5),
        ([-0.1, 1.1, 0.5, 0.2, -0.2, -0.5], 3.5),
    ],
    "linear3_arp": [
        ([0.5, 0.5], 0.5),
        ([1.5, 0.5], 1.5),
        ([0.9, -0.2, 0.8, -0.5], 2.5),
        ([0.9, 0.8, -0.6, 0.2, -0.5, -0.2, 0.4], 3.5),
    ],
}

NONLINEAR_GROUPS = {
    "nonlinear1_nma": [
        ([0.9, -0.2, 0.8, -0.5], 0.5),
        ([-0.3, 1.4, 0.4, -0.5], 1.5),
        ([1.5, -0.4, -0.3, 0.2], 2.5),
        ([-0.1, 1.4, 0.4, -0.7], 3.5),
    ],
    "nonlinear2_star1": [
        ([0.9, -0.2, 0.8, -0.5], 0.5),
        ([-0.3, 1.4, 0.4, -0.5], 1.5),
        ([1.5, -0.4, -0.3, 0.2], 2.5),
        ([-0.1, 1.4, 0.4, -0.7], 3.5),
    ],
    "nonlinear3_star2": [
        ([-0.5, 0.8, -0.2, 0.9], 0.5),
        ([-0.5, 0.4, 1.4, -0.3], 1.5),
        ([0.2, -0.3, -0.4, 1.5], 2.5),
        ([-0.7, 0.4, 1.4, -0.1], 3.5),
    ],
}

MODEL_KIND = {
    "linear1_ar4": "ar",
    "linear2_ar6": "ar",
    "linear3_arp": "ar",
    "nonlinear1_nma": "nma",
    "nonlinear2_star1": "star1",
    "nonlinear3_star2": "star2",
}


def blend_params(a: list[float], b: list[float], w: float) -> list[float]:
    """Linear interpolation, padding shorter AR orders with zeros."""
    max_len = max(len(a), len(b))
    aa = np.zeros(max_len)
    bb = np.zeros(max_len)
    aa[: len(a)] = a
    bb[: len(b)] = b
    return ((1 - w) * aa + w * bb).tolist()


def safe_gate(x: float) -> float:
    """Numerically stable version of (1 - exp(-10*x))^-1."""
    x = float(np.clip(x, -20, 20))
    denominator = 1.0 - math.exp(-10.0 * x)
    if abs(denominator) < 1e-8:
        denominator = 1e-8 if denominator >= 0 else -1e-8
    return 1.0 / denominator


def finite_value(x: float, limit: float = 1_000_000.0) -> float:
    """Avoid NaN/Inf caused by unstable printed parameter combinations."""
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, -limit, limit))


def generate_one_series(
    concepts: list[tuple[list[float], float]],
    model_kind: str,
    drift_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    max_p = max(len(params) for params, _ in concepts)
    history = list(rng.normal(0, 0.1, max(8, max_p + 2)))
    noises: list[float] = []
    values: list[float] = []
    concept_ids: list[int] = []

    for concept_idx, (params, sigma2) in enumerate(concepts):
        sigma = math.sqrt(sigma2)

        for j in range(CONCEPT_LEN):
            active_params = params
            active_sigma = sigma

            if drift_type == "gradual" and concept_idx > 0 and j < GRADUAL_LEN:
                previous_params, previous_sigma2 = concepts[concept_idx - 1]
                w = (j + 1) / GRADUAL_LEN
                active_params = blend_params(previous_params, params, w)
                active_sigma = (1 - w) * math.sqrt(previous_sigma2) + w * sigma

            if model_kind == "ar":
                lags = np.array([history[-i] for i in range(1, len(active_params) + 1)])
                x = float(np.dot(active_params, lags) + rng.normal(0, active_sigma))

            elif model_kind == "nma":
                # FEDD paper eq. 1:
                # x_t = w_t - a1*w_{t-1} + a2*w_{t-2}
                #       + a3*w_{t-1}w_{t-2} - a4*w_{t-2}^2
                a1, a2, a3, a4 = active_params
                wt = float(rng.normal(0, active_sigma))
                w1 = noises[-1] if len(noises) >= 1 else 0.0
                w2 = noises[-2] if len(noises) >= 2 else 0.0
                noises.append(wt)
                x = wt - a1 * w1 + a2 * w2 + a3 * w1 * w2 - a4 * (w2**2)

            elif model_kind == "star1":
                # FEDD paper eq. 2:
                # x_t = [a1*x_{t-1}+...+a4*x_{t-4}]
                #       * [1-exp(-10*x_{t-1})]^-1 + w_t
                h = np.array([history[-i] for i in range(1, 5)])
                x = float(np.dot(active_params, h) * safe_gate(history[-1]) + rng.normal(0, active_sigma))

            elif model_kind == "star2":
                # FEDD paper eq. 3:
                # x_t = a1*x_{t-1}+a2*x_{t-2}
                #       + [a3*x_{t-1}+a1*x_{t-2}]
                #       * [1-exp(-10*x_{t-1})]^-1 + w_t
                a1, a2, a3, _ = active_params
                x1, x2 = history[-1], history[-2]
                x = a1 * x1 + a2 * x2 + (a3 * x1 + a1 * x2) * safe_gate(x1) + rng.normal(0, active_sigma)

            else:
                raise ValueError(f"Unknown model kind: {model_kind}")

            x = finite_value(x)
            values.append(x)
            history.append(x)
            concept_ids.append(concept_idx + 1)

    return np.asarray(values, dtype=float), concept_ids


def generate_datasets(output_dir: Path, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_groups = {**LINEAR_GROUPS, **NONLINEAR_GROUPS}

    for group_name, concepts in all_groups.items():
        family = "linear" if group_name.startswith("linear") else "nonlinear"
        model_kind = MODEL_KIND[group_name]

        for drift_type in ["abrupt", "gradual"]:
            for rep in range(1, N_SERIES_PER_SUBSET + 1):
                values, concept_ids = generate_one_series(concepts, model_kind, drift_type, rng)
                file_name = f"{family}_{group_name}_{drift_type}_{rep:02d}.csv"

                pd.DataFrame(
                    {
                        "t": np.arange(1, TOTAL_LEN + 1),
                        "value": values,
                        "concept": concept_ids,
                    }
                ).to_csv(data_dir / file_name, index=False)

                rows.append(
                    {
                        "file": file_name,
                        "family": family,
                        "group": group_name,
                        "model": model_kind,
                        "drift_type": drift_type,
                        "length": TOTAL_LEN,
                        "concept_length": CONCEPT_LEN,
                        "drift_points": json.dumps(DRIFT_POINTS),
                        "gradual_transition_length": GRADUAL_LEN if drift_type == "gradual" else 0,
                        "concepts": json.dumps(
                            [{"parameters": params, "sigma2": sigma2} for params, sigma2 in concepts]
                        ),
                    }
                )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    return metadata


def run_fedd_detector(output_dir: Path, fedd_py_path: str, min_instances: int, warning: float, drift: float) -> pd.DataFrame:
    """
    Run the uploaded FEDD.py detector implementation.

    Requires dependencies used by that file, especially scikit-multiflow,
    statsmodels and scipy.
    """
    import importlib.util

    fedd_file = Path(fedd_py_path).resolve()
    spec = importlib.util.spec_from_file_location("uploaded_fedd", fedd_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    Detector = module.FeatureExtractionDriftDetector
    rows = []

    for csv_file in sorted((output_dir / "data").glob("*.csv")):
        detector = Detector(
            min_instances=min_instances,
            warning_threshold=warning,
            drift_threshold=drift,
        )
        values = pd.read_csv(csv_file)["value"].to_numpy()
        detections = []
        warnings = []

        for idx, x in enumerate(values, start=1):
            detector.add_element(float(x))
            if getattr(detector, "in_warning_zone", False):
                warnings.append(idx)
            if getattr(detector, "in_concept_change", False):
                detections.append(idx)

        rows.append(
            {
                "file": csv_file.name,
                "detections": json.dumps(detections),
                "n_detections": len(detections),
                "first_warning_indices_seen": json.dumps(warnings[:20]),
                "true_drift_points": json.dumps(DRIFT_POINTS),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "detection_results.csv", index=False)
    return results
