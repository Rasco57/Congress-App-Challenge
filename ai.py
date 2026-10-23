# Notice: All code below is written by ChatGPT. However, the code has been reviewed. All other code is written by me.

"""
baseball_rates.py

Calculate daily and cumulative SLG, OBP, and OPS from a pandas DataFrame,
then plot cumulative OPS with Plotly hover details.

Expected input:
    One row per plate appearance, with the listed event columns and a date
    column. Event columns should contain 1/0 or True/False indicators.

Usage:
    from baseball_rates import calculate_rates, plot_ops

    rates = calculate_rates(df)
    fig = plot_ops(rates)
    fig.show()

If the input has multiple rows per event/date, each row is treated as one
plate appearance/event record. The module sums the event indicators per day.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px


EVENT_COLUMNS = [
    "field_out",
    "single",
    "double",
    "walk",
    "strikeout",
    "home_run",
    "field_error",
    "grounded_into_double_play",
    "triple",
    "sac_fly",
    "force_out",
    "intent_walk",
    "fielders_choice",
    "hit_by_pitch",
    "strikeout_double_play",
    "truncated_pa",
    "date",
]

HIT_COLUMNS = ["single", "double", "triple", "home_run"]
TOTAL_BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4,
}

# Plate appearances that count toward OBP's denominator, apart from the
# hit-by-pitch and walk categories already included:
# AB + BB + HBP + SF
# AB = hits + ordinary outs + reached-on-error/fielders-choice/etc.
AT_BAT_OUTCOMES = [
    "field_out",
    "single",
    "double",
    "strikeout",
    "home_run",
    "field_error",
    "grounded_into_double_play",
    "triple",
    "force_out",
    "fielders_choice",
    "strikeout_double_play",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide series, returning NaN where denominator is zero."""
    return numerator.div(denominator.where(denominator != 0))


def calculate_rates(
    df: pd.DataFrame,
    date_column: str = "date",
    cumulative: bool = True,
) -> pd.DataFrame:
    """
    Return one row per date with daily and cumulative baseball rate stats.

    Rates:
    SLG = total bases / at-bats
    OBP = (hits + walks + hit-by-pitches) /
            (at-bats + walks + hit-by-pitches + sacrifice flies)
    OPS = OBP + SLG

    Parameters
    ----------
    df:
        DataFrame containing the event columns. Column order does not matter.
        Event columns should be indicator values (1/0 or True/False).
    date_column:
        Name of the date column.
    cumulative:
        If True (default), include cumulative counts and rates alongside daily
        rates. Cumulative values are calculated chronologically.

    Notes
    -----
    Missing event columns are treated as zero. Missing dates are dropped.
    ``truncated_pa`` is not added to any formula because it does not identify
    a completed outcome. It is still accepted as an input column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if date_column not in df.columns:
        raise ValueError(f"Required date column {date_column!r} was not found.")

    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data = data.dropna(subset=[date_column])

    if data.empty:
        raise ValueError("No valid dates were found in the DataFrame.")

    # Treat event indicators as numeric counts; absent columns mean zero.
    event_cols = [c for c in EVENT_COLUMNS if c != "date"]
    for col in event_cols:
        if col not in data.columns:
            data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data["hits"] = data[HIT_COLUMNS].sum(axis=1)
    data["total_bases"] = sum(
        data[col] * bases for col, bases in TOTAL_BASES.items()
    )
    data["at_bats"] = data[AT_BAT_OUTCOMES].sum(axis=1)

    # A walk and an intentional walk are separate event labels in some data
    # sources, but intentional walks are usually also represented as walks.
    # Avoid double-counting if both columns are populated on the same row.
    data["walks"] = data[["walk", "intent_walk"]].max(axis=1)
    data["hit_by_pitches"] = data["hit_by_pitch"]
    data["sac_flies"] = data["sac_fly"]

    grouped = data.groupby(data[date_column].dt.normalize(), sort=True)
    daily = grouped.agg(
        hits=("hits", "sum"),
        total_bases=("total_bases", "sum"),
        at_bats=("at_bats", "sum"),
        walks=("walks", "sum"),
        hit_by_pitches=("hit_by_pitches", "sum"),
        sac_flies=("sac_flies", "sum"),
    ).rename_axis("date").reset_index()

    daily["SLG"] = _safe_divide(daily["total_bases"], daily["at_bats"])
    daily["OBP"] = _safe_divide(
        daily["hits"] + daily["walks"] + daily["hit_by_pitches"],
        daily["at_bats"] + daily["walks"] + daily["hit_by_pitches"] + daily["sac_flies"],
    )
    daily["OPS"] = daily["OBP"] + daily["SLG"]

    if cumulative:
        count_columns = [
            "hits",
            "total_bases",
            "at_bats",
            "walks",
            "hit_by_pitches",
            "sac_flies",
        ]
        for col in count_columns:
            daily[f"cumulative_{col}"] = daily[col].cumsum()

        daily["cumulative_SLG"] = _safe_divide(
            daily["cumulative_total_bases"], daily["cumulative_at_bats"]
        )
        daily["cumulative_OBP"] = _safe_divide(
            daily["cumulative_hits"]
            + daily["cumulative_walks"]
            + daily["cumulative_hit_by_pitches"],
            daily["cumulative_at_bats"]
            + daily["cumulative_walks"]
            + daily["cumulative_hit_by_pitches"]
            + daily["cumulative_sac_flies"],
        )
        daily["cumulative_OPS"] = (
            daily["cumulative_OBP"] + daily["cumulative_SLG"]
        )

    return daily


def plot_ops(
    rates: pd.DataFrame,
    cumulative: bool = True,
    show_daily_rates: bool = False,
    title: Optional[str] = None,
):
    """
    Create an interactive Plotly line graph.

    Parameters
    ----------
    rates:
        Output from ``calculate_rates``.
    cumulative:
        If True, graph cumulative OPS; otherwise graph daily OPS.
    show_daily_rates:
        If True, graph daily SLG, OBP, and OPS as three lines instead of OPS
        alone. When cumulative=True, this option graphs cumulative SLG, OBP,
        and OPS.
    title:
        Optional chart title.
    """
    if "date" not in rates.columns:
        raise ValueError("rates must contain a 'date' column.")

    if show_daily_rates:
        prefix = "cumulative_" if cumulative else ""
        y_columns = [f"{prefix}SLG", f"{prefix}OBP", f"{prefix}OPS"]
        missing = [col for col in y_columns if col not in rates.columns]
        if missing:
            raise ValueError(f"Missing rate columns: {', '.join(missing)}")

        plot_data = rates[["date"] + y_columns].melt(
            id_vars="date", var_name="Statistic", value_name="Rate"
        )
        plot_data["Statistic"] = plot_data["Statistic"].str.replace(
            "cumulative_", "", regex=False
        )
        fig = px.line(
            plot_data,
            x="date",
            y="Rate",
            color="Statistic",
            markers=True,
            title=title or (
                "Cumulative SLG, OBP, and OPS"
                if cumulative else "Daily SLG, OBP, and OPS"
            ),
            hover_data={"date": "|%Y-%m-%d", "Rate": ":.3f", "Statistic": True},
        )
        fig.update_traces(hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Statistic: %{fullData.name}<br>"
            "Value: %{y:.3f}<extra></extra>"
        ))
    else:
        ops_col = "cumulative_OPS" if cumulative else "OPS"
        slg_col = "cumulative_SLG" if cumulative else "SLG"
        obp_col = "cumulative_OBP" if cumulative else "OBP"
        for col in (ops_col, slg_col, obp_col):
            if col not in rates.columns:
                raise ValueError(f"Missing rate column {col!r}.")

        plot_data = rates.copy()
        fig = px.line(
            plot_data,
            x="date",
            y=ops_col,
            markers=True,
            title=title or ("Baseball Stocks" if cumulative else "Daily OPS by Date"),
            custom_data=[slg_col, obp_col],
        )
        fig.update_traces(
            name="OPS",
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "OPS: %{y:.3f}<br>"
                "SLG: %{customdata[0]:.3f}<br>"
                "OBP: %{customdata[1]:.3f}"
                "<extra></extra>"
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Rate",
        hovermode="x unified" if show_daily_rates else "closest",
        template="plotly_dark",
    )
    return fig
