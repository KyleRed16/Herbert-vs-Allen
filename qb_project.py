
import pandas as pd
import numpy as np


def build_qb_dataset(pbp, qb_name):

    qb_plays = pbp[
        (pbp["passer_player_name"] == qb_name) |
        (pbp["rusher_player_name"] == qb_name)
    ].copy()

    qb_plays["is_dropback"] = (
        qb_plays["qb_dropback"] == 1
    )

    qb_plays["is_scramble"] = (
        (qb_plays["play_type"] == "run") &
        (qb_plays["qb_scramble"] == 1)
    )

    qb_plays["is_designed_run"] = (
        (qb_plays["play_type"] == "run") &
        (qb_plays["rusher_player_name"] == qb_name) &
        (qb_plays["qb_scramble"] == 0) &
        (qb_plays["rush_attempt"] == 1)
    )

    qb_plays["include_in_analysis"] = (
        qb_plays["is_dropback"] |
        qb_plays["is_scramble"] |
        qb_plays["is_designed_run"]
    )

    analysis_plays = qb_plays[
        qb_plays["include_in_analysis"]
    ].copy()

    return analysis_plays

def bootstrap_epa_gap(
    herbert_games,
    allen_games,
    iterations=10000,
    seed=42
):
    rng = np.random.default_rng(seed)

    bootstrap_gaps = np.empty(iterations)

    for i in range(iterations):

        herbert_sample = herbert_games.iloc[
            rng.integers(
                0,
                len(herbert_games),
                size=len(herbert_games)
            )
        ]

        allen_sample = allen_games.iloc[
            rng.integers(
                0,
                len(allen_games),
                size=len(allen_games)
            )
        ]

        herbert_epa = (
            herbert_sample["total_epa"].sum() /
            herbert_sample["total_plays"].sum()
        )

        allen_epa = (
            allen_sample["total_epa"].sum() /
            allen_sample["total_plays"].sum()
        )

        bootstrap_gaps[i] = allen_epa - herbert_epa

    observed_gap = (
        allen_games["total_epa"].sum() /
        allen_games["total_plays"].sum()
    ) - (
        herbert_games["total_epa"].sum() /
        herbert_games["total_plays"].sum()
    )

    return {
        "observed_gap": observed_gap,
        "lower_95": np.percentile(bootstrap_gaps, 2.5),
        "upper_95": np.percentile(bootstrap_gaps, 97.5),
        "bootstrap_gaps": bootstrap_gaps
    }
