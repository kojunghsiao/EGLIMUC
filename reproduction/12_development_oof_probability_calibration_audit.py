# Exact function bodies extracted from the verified saved-prediction audit sources.
# Loaded by reproduce.py; this module is not a standalone workflow.

def equal_width_bin_ids(
    probabilities,
    n_bins,
):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    bin_ids = np.floor(
        probabilities
        *
        n_bins
    ).astype(int)

    return np.clip(
        bin_ids,
        0,
        n_bins - 1,
    )

def weighted_mean(
    values,
    weights,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    total_weight = float(
        weights.sum()
    )

    if total_weight <= 0:
        return np.nan

    return float(
        np.sum(
            values
            *
            weights
        )
        /
        total_weight
    )

def build_calibration_precomputation(
    y_true,
    probabilities,
    n_bins,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        PROBABILITY_EPSILON,
        1.0,
    )

    probabilities = (
        probabilities
        /
        probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    one_hot = np.eye(
        NUM_CLASSES,
        dtype=float,
    )[
        y_true
    ]

    predicted_class = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    correct = (
        predicted_class
        ==
        y_true
    ).astype(float)

    nll_per_observation = -np.log(
        probabilities[
            np.arange(
                len(y_true)
            ),
            y_true,
        ]
    )

    brier_per_observation = np.sum(
        (
            probabilities
            -
            one_hot
        )
        ** 2,
        axis=1,
    )

    cumulative_probabilities = np.cumsum(
        probabilities[
            :,
            :-1,
        ],
        axis=1,
    )

    cumulative_outcomes = np.column_stack(
        [
            (
                y_true
                <=
                threshold
            ).astype(float)
            for threshold in range(
                NUM_CLASSES - 1
            )
        ]
    )

    ranked_probability_per_observation = (
        np.sum(
            (
                cumulative_probabilities
                -
                cumulative_outcomes
            )
            ** 2,
            axis=1,
        )
        /
        (
            NUM_CLASSES - 1
        )
    )

    top_bin_ids = equal_width_bin_ids(
        confidence,
        n_bins,
    )

    class_bin_ids = np.column_stack(
        [
            equal_width_bin_ids(
                probabilities[
                    :,
                    class_id,
                ],
                n_bins,
            )
            for class_id in CLASS_IDS
        ]
    )

    class_targets = np.column_stack(
        [
            (
                y_true
                ==
                class_id
            ).astype(float)
            for class_id in CLASS_IDS
        ]
    )

    return {
        "y_true": y_true,
        "probabilities": probabilities,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "correct": correct,
        "nll_per_observation": (
            nll_per_observation
        ),
        "brier_per_observation": (
            brier_per_observation
        ),
        "rps_per_observation": (
            ranked_probability_per_observation
        ),
        "top_bin_ids": top_bin_ids,
        "class_bin_ids": class_bin_ids,
        "class_targets": class_targets,
        "n_bins": int(
            n_bins
        ),
    }

def binned_calibration_statistics(
    bin_ids,
    predicted_probability,
    observed_outcome,
    weights,
    n_bins,
):
    bin_ids = np.asarray(
        bin_ids,
        dtype=int,
    )

    predicted_probability = np.asarray(
        predicted_probability,
        dtype=float,
    )

    observed_outcome = np.asarray(
        observed_outcome,
        dtype=float,
    )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    bin_weight = np.bincount(
        bin_ids,
        weights=weights,
        minlength=n_bins,
    ).astype(float)

    probability_sum = np.bincount(
        bin_ids,
        weights=(
            weights
            *
            predicted_probability
        ),
        minlength=n_bins,
    ).astype(float)

    outcome_sum = np.bincount(
        bin_ids,
        weights=(
            weights
            *
            observed_outcome
        ),
        minlength=n_bins,
    ).astype(float)

    mean_probability = np.divide(
        probability_sum,
        bin_weight,
        out=np.full(
            n_bins,
            np.nan,
            dtype=float,
        ),
        where=bin_weight > 0,
    )

    observed_frequency = np.divide(
        outcome_sum,
        bin_weight,
        out=np.full(
            n_bins,
            np.nan,
            dtype=float,
        ),
        where=bin_weight > 0,
    )

    absolute_gap = np.abs(
        observed_frequency
        -
        mean_probability
    )

    valid_bins = (
        bin_weight > 0
    )

    total_weight = float(
        bin_weight.sum()
    )

    if total_weight <= 0:
        return {
            "bin_weight": bin_weight,
            "mean_probability": mean_probability,
            "observed_frequency": observed_frequency,
            "absolute_gap": absolute_gap,
            "ece": np.nan,
            "mce": np.nan,
            "rmsce": np.nan,
        }

    normalized_bin_weight = (
        bin_weight
        /
        total_weight
    )

    ece = float(
        np.nansum(
            normalized_bin_weight
            *
            absolute_gap
        )
    )

    mce = float(
        np.nanmax(
            absolute_gap[
                valid_bins
            ]
        )
    )

    rmsce = float(
        np.sqrt(
            np.nansum(
                normalized_bin_weight
                *
                (
                    absolute_gap
                    ** 2
                )
            )
        )
    )

    return {
        "bin_weight": bin_weight,
        "mean_probability": mean_probability,
        "observed_frequency": observed_frequency,
        "absolute_gap": absolute_gap,
        "ece": ece,
        "mce": mce,
        "rmsce": rmsce,
    }

def calibration_metrics_from_precomputation(
    precomputed,
    sample_weights=None,
):
    n_observations = len(
        precomputed[
            "y_true"
        ]
    )

    if sample_weights is None:
        sample_weights = np.ones(
            n_observations,
            dtype=float,
        )

    else:
        sample_weights = np.asarray(
            sample_weights,
            dtype=float,
        )

    total_weight = float(
        sample_weights.sum()
    )

    if total_weight <= 0:
        raise ValueError(
            "Calibration sample weights sum to zero."
        )

    nll = weighted_mean(
        precomputed[
            "nll_per_observation"
        ],
        sample_weights,
    )

    brier = weighted_mean(
        precomputed[
            "brier_per_observation"
        ],
        sample_weights,
    )

    ranked_probability_score = weighted_mean(
        precomputed[
            "rps_per_observation"
        ],
        sample_weights,
    )

    accuracy = weighted_mean(
        precomputed[
            "correct"
        ],
        sample_weights,
    )

    mean_confidence = weighted_mean(
        precomputed[
            "confidence"
        ],
        sample_weights,
    )

    top_statistics = binned_calibration_statistics(
        bin_ids=precomputed[
            "top_bin_ids"
        ],
        predicted_probability=precomputed[
            "confidence"
        ],
        observed_outcome=precomputed[
            "correct"
        ],
        weights=sample_weights,
        n_bins=precomputed[
            "n_bins"
        ],
    )

    classwise_ece_values = []

    for class_id in CLASS_IDS:
        class_statistics = (
            binned_calibration_statistics(
                bin_ids=precomputed[
                    "class_bin_ids"
                ][
                    :,
                    class_id,
                ],
                predicted_probability=precomputed[
                    "probabilities"
                ][
                    :,
                    class_id,
                ],
                observed_outcome=precomputed[
                    "class_targets"
                ][
                    :,
                    class_id,
                ],
                weights=sample_weights,
                n_bins=precomputed[
                    "n_bins"
                ],
            )
        )

        classwise_ece_values.append(
            class_statistics[
                "ece"
            ]
        )

    output = {
        "nll": float(
            nll
        ),
        "multiclass_brier": float(
            brier
        ),
        "ranked_probability_score": float(
            ranked_probability_score
        ),
        "top_label_ece": float(
            top_statistics[
                "ece"
            ]
        ),
        "top_label_mce": float(
            top_statistics[
                "mce"
            ]
        ),
        "top_label_rmsce": float(
            top_statistics[
                "rmsce"
            ]
        ),
        "classwise_ece_macro": float(
            np.mean(
                classwise_ece_values
            )
        ),
        "accuracy": float(
            accuracy
        ),
        "mean_confidence": float(
            mean_confidence
        ),
        "confidence_minus_accuracy": float(
            mean_confidence
            -
            accuracy
        ),
    }

    for class_id in CLASS_IDS:
        output[
            f"classwise_ece_{class_id}"
        ] = float(
            classwise_ece_values[
                class_id
            ]
        )

    return output
