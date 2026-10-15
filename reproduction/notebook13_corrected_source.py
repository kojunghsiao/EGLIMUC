# Exact function bodies extracted from the verified saved-prediction audit sources.
# Loaded by reproduce.py; this module is not a standalone workflow.

def safe_qwk(
    y_true,
    y_pred,
):

    try:

        value = cohen_kappa_score(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            weights="quadratic",
        )


        if not np.isfinite(
            value
        ):

            return np.nan


        return float(
            value
        )


    except Exception:

        return np.nan

def fixed_balanced_accuracy(
    y_true,
    y_pred,
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_IDS,
    ).astype(
        float
    )


    row_sums = matrix.sum(
        axis=1
    )


    recalls = np.divide(
        np.diag(
            matrix
        ),
        row_sums,
        out=np.zeros(
            NUM_CLASSES,
            dtype=float,
        ),
        where=(
            row_sums
            >
            0
        ),
    )


    present_classes = (
        row_sums
        >
        0
    )


    if not present_classes.any():

        return np.nan


    return float(
        recalls[
            present_classes
        ].mean()
    )

def compute_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )


    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )


    return {
        "qwk": safe_qwk(
            y_true,
            y_pred,
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": (
            fixed_balanced_accuracy(
                y_true,
                y_pred,
            )
        ),
    }

def normalize_probability_vector(
    vector,
):

    vector = np.asarray(
        vector,
        dtype=float,
    )


    vector = np.clip(
        vector,
        0.0,
        None,
    )


    total = float(
        vector.sum()
    )


    if total <= 0:

        return (
            np.ones(
                NUM_CLASSES,
                dtype=float,
            )
            /
            NUM_CLASSES
        )


    return (
        vector
        /
        total
    )

def round_half_up(
    value,
):

    return int(
        math.floor(
            float(
                value
            )
            +
            0.5
        )
    )

def round_half_up_to_class(
    value,
):

    return int(
        np.clip(
            round_half_up(
                value
            ),
            0,
            NUM_CLASSES
            -
            1,
        )
    )

def aggregate_one_patient(
    patient_group,
    aggregation_rule,
):

    probabilities = (
        patient_group[
            PROBABILITY_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    hard_predictions = np.argmax(
        probabilities,
        axis=1,
    )


    patient_truth_values = (
        patient_group[
            "patient_y_true"
        ]
        .astype(
            int
        )
        .unique()
    )


    if len(
        patient_truth_values
    ) != 1:

        raise RuntimeError(
            "A patient has inconsistent derived reference labels."
        )


    mean_probabilities = (
        normalize_probability_vector(
            probabilities.mean(
                axis=0
            )
        )
    )


    if (
        aggregation_rule
        ==
        "mean_probability_argmax"
    ):

        aggregate_probabilities = (
            mean_probabilities
        )

        patient_prediction = int(
            np.argmax(
                aggregate_probabilities
            )
        )


    elif (
        aggregation_rule
        ==
        "median_probability_argmax"
    ):

        aggregate_probabilities = (
            normalize_probability_vector(
                np.median(
                    probabilities,
                    axis=0,
                )
            )
        )


        patient_prediction = int(
            np.argmax(
                aggregate_probabilities
            )
        )


    elif (
        aggregation_rule
        ==
        "majority_vote_low_tie"
    ):

        vote_counts = np.bincount(
            hard_predictions,
            minlength=NUM_CLASSES,
        ).astype(
            float
        )


        tied_classes = np.flatnonzero(
            vote_counts
            ==
            vote_counts.max()
        )


        patient_prediction = int(
            tied_classes[
                0
            ]
        )


        aggregate_probabilities = (
            normalize_probability_vector(
                vote_counts
            )
        )


    elif (
        aggregation_rule
        ==
        "mean_expected_mes_round"
    ):

        expected_mes = (
            probabilities
            @
            np.arange(
                NUM_CLASSES,
                dtype=float,
            )
        )


        patient_prediction = (
            round_half_up_to_class(
                expected_mes.mean()
            )
        )


        aggregate_probabilities = np.zeros(
            NUM_CLASSES,
            dtype=float,
        )


        aggregate_probabilities[
            patient_prediction
        ] = 1.0


    elif (
        aggregation_rule
        ==
        "median_expected_mes_round"
    ):

        expected_mes = (
            probabilities
            @
            np.arange(
                NUM_CLASSES,
                dtype=float,
            )
        )


        patient_prediction = (
            round_half_up_to_class(
                np.median(
                    expected_mes
                )
            )
        )


        aggregate_probabilities = np.zeros(
            NUM_CLASSES,
            dtype=float,
        )


        aggregate_probabilities[
            patient_prediction
        ] = 1.0


    elif (
        aggregation_rule
        ==
        "maximum_predicted_image_class"
    ):

        patient_prediction = int(
            hard_predictions.max()
        )


        aggregate_probabilities = np.zeros(
            NUM_CLASSES,
            dtype=float,
        )


        aggregate_probabilities[
            patient_prediction
        ] = 1.0


    else:

        raise ValueError(
            f"Unknown aggregation rule: {aggregation_rule}"
        )


    output_row = {
        "patient_id": int(
            patient_group[
                "patient_id"
            ].iloc[
                0
            ]
        ),
        "y_true": int(
            patient_truth_values[
                0
            ]
        ),
        "y_pred": int(
            patient_prediction
        ),
        "n_images": int(
            len(
                patient_group
            )
        ),
        "aggregation_rule": (
            aggregation_rule
        ),
    }


    if "outer_fold" in patient_group.columns:

        values = (
            patient_group[
                "outer_fold"
            ]
            .dropna()
            .astype(
                int
            )
            .unique()
        )


        if len(
            values
        ) == 1:

            output_row[
                "outer_fold"
            ] = int(
                values[
                    0
                ]
            )


    for class_id, probability_column in enumerate(
        PROBABILITY_COLUMNS
    ):

        output_row[
            probability_column
        ] = float(
            aggregate_probabilities[
                class_id
            ]
        )


    return output_row

def aggregate_image_predictions(
    image_dataframe,
    aggregation_rule,
):

    rows = []


    for _, patient_group in (
        image_dataframe.groupby(
            "patient_id",
            sort=True,
        )
    ):

        rows.append(
            aggregate_one_patient(
                patient_group,
                aggregation_rule,
            )
        )


    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "patient_id"
        )
        .reset_index(
            drop=True
        )
    )

def normalize_path(
    value,
):

    return re.sub(
        r"/+",
        "/",
        str(
            value
        )
        .replace(
            "\\",
            "/",
        )
        .strip(),
    )

def canonical_integer_column(
    dataframe,
    column,
):

    dataframe[
        column
    ] = pd.to_numeric(
        dataframe[
            column
        ],
        errors="raise",
    )


    numeric_values = dataframe[
        column
    ].to_numpy(
        dtype=float
    )


    if not np.allclose(
        numeric_values,
        np.round(
            numeric_values
        ),
    ):

        raise RuntimeError(
            f"{column} contains non-integer values."
        )


    dataframe[
        column
    ] = dataframe[
        column
    ].astype(
        int
    )

def attach_labels(
    blind_dataframe,
):

    blind_dataframe = (
        blind_dataframe.copy()
    )


    forbidden_columns = {
        "label",
        "y_true",
        "patient_y_true",
    }


    present_forbidden = (
        forbidden_columns
        &
        set(
            blind_dataframe.columns
        )
    )


    if present_forbidden:

        raise RuntimeError(
            "Input outer prediction dataframe is not label blind.\n"
            f"Unexpected columns: {sorted(present_forbidden)}"
        )


    required_columns = {
        "relative_path",
        "patient_id",
        "cv_fold",
        "y_pred",
        *PROBABILITY_COLUMNS,
    }


    missing_columns = (
        required_columns
        -
        set(
            blind_dataframe.columns
        )
    )


    if missing_columns:

        raise RuntimeError(
            "Blind prediction dataframe missing columns:\n"
            f"{sorted(missing_columns)}\n\n"
            f"Available columns:\n"
            f"{list(blind_dataframe.columns)}"
        )


    blind_dataframe[
        "relative_path"
    ] = blind_dataframe[
        "relative_path"
    ].astype(
        str
    ).map(
        normalize_path
    )


    for integer_column in [
        "patient_id",
        "cv_fold",
        "y_pred",
    ]:

        canonical_integer_column(
            blind_dataframe,
            integer_column,
        )


    for probability_column in PROBABILITY_COLUMNS:

        blind_dataframe[
            probability_column
        ] = pd.to_numeric(
            blind_dataframe[
                probability_column
            ],
            errors="raise",
        ).astype(
            float
        )


    if blind_dataframe[
        "relative_path"
    ].duplicated().any():

        raise RuntimeError(
            "Blind prediction file contains duplicate relative paths."
        )


    probability_array = blind_dataframe[
        PROBABILITY_COLUMNS
    ].to_numpy(
        dtype=float
    )


    if not np.isfinite(
        probability_array
    ).all():

        raise RuntimeError(
            "Blind prediction file contains non-finite probabilities."
        )


    vault_for_join = (
        outer_label_vault_df.rename(
            columns={
                "patient_id": (
                    "_vault_patient_id"
                ),
                "cv_fold": (
                    "_vault_cv_fold"
                ),
            }
        )
        .copy()
    )


    input_row_count = len(
        blind_dataframe
    )


    merged = blind_dataframe.merge(
        vault_for_join,
        on="relative_path",
        how="inner",
        validate="one_to_one",
    )


    if len(
        merged
    ) != input_row_count:

        unmatched_paths = sorted(
            set(
                blind_dataframe[
                    "relative_path"
                ]
            )
            -
            set(
                merged[
                    "relative_path"
                ]
            )
        )


        print(
            "\nExample unmatched paths:"
        )


        print(
            unmatched_paths[
                :20
            ]
        )


        raise RuntimeError(
            "Final label join changed the number of prediction rows.\n"
            f"Before: {input_row_count}\n"
            f"After: {len(merged)}"
        )


    patient_id_mismatch = (
        merged[
            "patient_id"
        ].astype(
            int
        )
        !=
        merged[
            "_vault_patient_id"
        ].astype(
            int
        )
    )


    if patient_id_mismatch.any():

        display(
            merged.loc[
                patient_id_mismatch,
                [
                    "relative_path",
                    "patient_id",
                    "_vault_patient_id",
                ],
            ].head(
                20
            )
        )


        raise RuntimeError(
            "Blind prediction patient_id disagrees with frozen manifest."
        )


    fold_mismatch = (
        merged[
            "cv_fold"
        ].astype(
            int
        )
        !=
        merged[
            "_vault_cv_fold"
        ].astype(
            int
        )
    )


    if fold_mismatch.any():

        display(
            merged.loc[
                fold_mismatch,
                [
                    "relative_path",
                    "patient_id",
                    "cv_fold",
                    "_vault_cv_fold",
                ],
            ].head(
                20
            )
        )


        raise RuntimeError(
            "Blind prediction cv_fold disagrees with frozen manifest."
        )


    merged[
        "y_true"
    ] = merged[
        "label"
    ].astype(
        int
    )


    merged[
        "patient_y_true"
    ] = merged[
        "patient_y_true"
    ].astype(
        int
    )


    merged = merged.drop(
        columns=[
            "_vault_patient_id",
            "_vault_cv_fold",
        ]
    )


    forbidden_suffix_columns = {
        "patient_id_x",
        "patient_id_y",
        "cv_fold_x",
        "cv_fold_y",
        "patient_y_true_x",
        "patient_y_true_y",
    }


    unexpected_suffix_columns = (
        forbidden_suffix_columns
        &
        set(
            merged.columns
        )
    )


    if unexpected_suffix_columns:

        raise RuntimeError(
            "Unexpected merge suffix columns remain:\n"
            f"{sorted(unexpected_suffix_columns)}"
        )


    if "patient_id" not in merged.columns:

        raise RuntimeError(
            "patient_id disappeared during final label attachment."
        )


    if "cv_fold" not in merged.columns:

        raise RuntimeError(
            "cv_fold disappeared during final label attachment."
        )


    if merged[
        "patient_id"
    ].isna().any():

        raise RuntimeError(
            "Missing patient IDs after label attachment."
        )


    if merged[
        "y_true"
    ].isna().any():

        raise RuntimeError(
            "Missing image labels after label attachment."
        )


    if merged[
        "patient_y_true"
    ].isna().any():

        raise RuntimeError(
            "Missing patient references after label attachment."
        )


    return merged

def stratified_bootstrap(
    patient_dataframe,
    repetitions,
    seed,
):

    dataframe = (
        patient_dataframe.sort_values(
            "patient_id"
        )
        .reset_index(
            drop=True
        )
    )


    y_true = dataframe[
        "y_true"
    ].to_numpy(
        dtype=int
    )


    y_pred = dataframe[
        "y_pred"
    ].to_numpy(
        dtype=int
    )


    class_indices = {
        class_id: np.flatnonzero(
            y_true
            ==
            class_id
        )
        for class_id in CLASS_IDS
    }


    rng = np.random.default_rng(
        seed
    )


    rows = []


    for bootstrap_index in range(
        repetitions
    ):

        sampled = np.concatenate(
            [
                rng.choice(
                    class_indices[
                        class_id
                    ],
                    size=len(
                        class_indices[
                            class_id
                        ]
                    ),
                    replace=True,
                )
                for class_id in CLASS_IDS
            ]
        )


        metrics = compute_metrics(
            y_true[
                sampled
            ],
            y_pred[
                sampled
            ],
        )


        rows.append(
            {
                "bootstrap_index": (
                    bootstrap_index
                ),
                **metrics,
            }
        )


    return pd.DataFrame(
        rows
    )
