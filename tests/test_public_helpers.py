"""Synthetic checks only; no data, network, training, or workflow execution.
Run: python tests/test_public_helpers.py
Requires the CPU reproduction dependencies.
"""
import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, mean_absolute_error, f1_score,
    accuracy_score, confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
NUM_CLASSES = 4
CLASS_IDS = list(range(4))
PROBABILITY_COLUMNS = [f"prob_{i}" for i in CLASS_IDS]
PROBABILITY_EPSILON = 1e-12

for name in [
    "notebook13_corrected_source.py",
    "12_development_oof_probability_calibration_audit.py",
]:
    source = ROOT / "reproduction" / name
    tree = ast.parse(source.read_text())
    # Only pure function definitions are loaded; no workflow top-level calls.
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), name, "exec"), globals())

assert round_half_up(2.5) == 3
assert abs(safe_qwk([0, 1, 2, 3], [0, 1, 2, 3]) - 1) < 1e-15
for labels, probabilities in [
    ([0, 1, 2, 3], np.eye(4)),
    ([0, 0, 1, 1], np.full((4, 4), 0.25)),
]:
    values = calibration_metrics_from_precomputation(
        build_calibration_precomputation(labels, probabilities, 15)
    )
    assert all(np.isfinite(value) for value in values.values())
print("PASS: synthetic rounding, QWK, and 15-bin calibration checks. No study data tested.")
