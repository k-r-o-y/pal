#!/usr/bin/env python3
"""
Publication-quality plotting for the constrained-polytope benchmark.

Run from the repository root:

    python -m analysis.plots.plot_polytope_benchmark --format pdf png svg

Expected input:
    results/polytope/polytope_results.csv

Generated figures (when columns are available):
    polytope_integration_error_vs_degree.*
    polytope_condition_number_vs_degree.*
    polytope_rank_fraction_vs_degree.*
    polytope_coefficient_noise_amplification_vs_degree.*
    polytope_recovered_integral_error_vs_degree.*
    polytope_perturbation_sensitivity_vs_degree.*
    polytope_runtime_vs_degree.*
    polytope_dynamic_schedule_error.*
    polytope_dynamic_trajectory_error.*
    polytope_error_summary.*
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS_PATH = Path("results/polytope/polytope_results.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("figures")
DEFAULT_SUMMARY_PATH = Path("results/polytope/polytope_plot_summary.json")
DEFAULT_FORMATS = ("pdf", "png", "svg")
SUPPORTED_FORMATS = DEFAULT_FORMATS

BASIS_ORDER = ("monomial", "legendre", "chebyshev")
DTYPE_ORDER = ("float32", "float64")
SCENARIO_ORDER = ("convex_box", "box_with_obstacle", "dynamic_box_with_obstacle")
SCHEDULE_ORDER = ("static", "shrinking", "expanding", "oscillating", "pulsed", "random_walk")

BASIS_DISPLAY = {"monomial": "Monomial", "legendre": "Legendre", "chebyshev": "Chebyshev"}
DTYPE_DISPLAY = {"float32": "float32", "float64": "float64"}
SCENARIO_DISPLAY = {
    "convex_box": "Convex box",
    "box_with_obstacle": "Box with obstacle",
    "dynamic_box_with_obstacle": "Dynamic box with obstacle",
}
SCHEDULE_DISPLAY = {
    "static": "Static",
    "shrinking": "Shrinking",
    "expanding": "Expanding",
    "oscillating": "Oscillating",
    "pulsed": "Pulsed",
    "random_walk": "Random walk",
}
MARKERS = {"monomial": "o", "legendre": "s", "chebyshev": "^"}
LINESTYLES = {"float32": "-", "float64": "--"}

BASE_REQUIRED_COLUMNS = {
    "scenario", "basis", "dtype", "dimension", "source_polynomial_degree",
    "density_polynomial_degree", "trial", "relative_integration_error",
    "perturbation_sensitivity", "total_evaluation_ms",
}
OPTIONAL_NUMERIC_COLUMNS = {
    "trajectory_step", "obstacle_half_width", "reference_integral", "computed_integral",
    "basis_condition_number", "numerical_rank_fraction", "function_value_noise_relative_norm",
    "coefficient_recovery_relative_error", "coefficient_noise_amplification",
    "recovered_integral_relative_error", "basis_conversion_residual",
    "sampled_minimum_density_value",
}
METRIC_LABELS = {
    "relative_integration_error": "Relative integration error",
    "basis_condition_number": "Basis condition number",
    "numerical_rank_fraction": "Retained numerical-rank fraction",
    "coefficient_noise_amplification": "Coefficient-noise amplification",
    "recovered_integral_relative_error": "Recovered-integral relative error",
    "perturbation_sensitivity": "Coefficient-perturbation sensitivity",
    "total_evaluation_ms": "Total evaluation runtime (ms)",
}


@dataclass(frozen=True)
class PlotContext:
    output_directory: Path
    formats: tuple[str, ...]
    dpi: int
    positive_floor: float
    figure_width: float
    figure_height: float
    detail_dimension: int
    detail_scenario: str
    detail_schedule: str
    legend_columns: int


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dissertation-quality plots from polytope_results.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--format", dest="formats", nargs="+", choices=SUPPORTED_FORMATS,
                        default=list(DEFAULT_FORMATS))
    parser.add_argument("--dimensions", nargs="+", type=int, default=None)
    parser.add_argument("--source-degrees", nargs="+", type=int, default=None)
    parser.add_argument("--density-degrees", nargs="+", type=int, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--schedules", nargs="+", default=None)
    parser.add_argument("--bases", nargs="+", choices=BASIS_ORDER, default=None)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPE_ORDER, default=None)
    parser.add_argument("--detail-dimension", type=int, default=None)
    parser.add_argument("--detail-scenario", default="dynamic_box_with_obstacle")
    parser.add_argument("--detail-schedule", default="oscillating")
    parser.add_argument("--positive-floor", type=float, default=1e-18)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-width", type=float, default=8.3)
    parser.add_argument("--figure-height", type=float, default=5.4)
    parser.add_argument("--legend-columns", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.positive_floor <= 0:
        parser.error("--positive-floor must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.figure_width <= 0 or args.figure_height <= 0:
        parser.error("figure dimensions must be positive")
    args.formats = tuple(dict.fromkeys(args.formats))
    return args


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results not found at {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Results file is empty: {path}")
    missing = sorted(BASE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = frame.copy()
    numeric_columns = (BASE_REQUIRED_COLUMNS | OPTIONAL_NUMERIC_COLUMNS |
                       {"dimension", "source_polynomial_degree", "density_polynomial_degree", "trial"})
    for column in sorted(numeric_columns & set(frame.columns)):
        if column not in {"scenario", "basis", "dtype"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("scenario", "basis", "dtype"):
        frame[column] = frame[column].astype(str).str.strip().str.lower()
    if "schedule" not in frame.columns:
        frame["schedule"] = np.where(frame["scenario"].eq("dynamic_box_with_obstacle"),
                                     "unknown", "static")
    else:
        frame["schedule"] = frame["schedule"].fillna("static").astype(str).str.strip().str.lower()
    if "trajectory_step" not in frame.columns:
        frame["trajectory_step"] = 0
    frame["is_finite_core"] = np.isfinite(frame[[
        "relative_integration_error", "perturbation_sensitivity", "total_evaluation_ms"
    ]].to_numpy(dtype=float)).all(axis=1)
    return frame


def apply_filters(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = frame.copy()
    for column, values in [
        ("dimension", args.dimensions),
        ("source_polynomial_degree", args.source_degrees),
        ("density_polynomial_degree", args.density_degrees),
        ("scenario", args.scenarios),
        ("schedule", args.schedules),
        ("basis", args.bases),
        ("dtype", args.dtypes),
    ]:
        if values is not None:
            filtered = filtered[filtered[column].isin(values)]
    if filtered.empty:
        raise ValueError("No rows remain after filtering")
    return filtered.reset_index(drop=True)


def ordered_present(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    present = list(dict.fromkeys(str(v) for v in values))
    result = [v for v in preferred if v in present]
    result.extend(v for v in present if v not in result)
    return result


def choose_detail_dimension(frame: pd.DataFrame, requested: int | None) -> int:
    dimensions = sorted(int(v) for v in frame["dimension"].dropna().unique())
    if not dimensions:
        raise ValueError("No valid dimensions")
    if requested is None:
        return dimensions[-1]
    if requested not in dimensions:
        raise ValueError(f"Dimension {requested} unavailable; available: {dimensions}")
    return requested


def choose_detail_value(frame: pd.DataFrame, column: str, requested: str,
                        preferred: Sequence[str], label: str) -> str:
    available = ordered_present(frame[column].unique(), preferred)
    if requested in available:
        return requested
    fallback = available[-1]
    warnings.warn(f"Requested {label} {requested!r} unavailable; using {fallback!r}", RuntimeWarning)
    return fallback


def finite_metric_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        return frame.iloc[0:0].copy()
    values = pd.to_numeric(frame[metric], errors="coerce")
    return frame[np.isfinite(values)].copy()


def aggregate_metric(frame: pd.DataFrame, groups: Sequence[str], metric: str) -> pd.DataFrame:
    usable = finite_metric_rows(frame, metric)
    if usable.empty:
        return pd.DataFrame(columns=[*groups, "median", "q1", "q3", "count"])
    return (usable.groupby(list(groups), observed=True, dropna=False)[metric]
            .agg(median="median", q1=lambda s: s.quantile(.25),
                 q3=lambda s: s.quantile(.75), count="count")
            .reset_index())


def safe_positive(values: Sequence[float] | np.ndarray, floor: float, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    zeros = int(np.count_nonzero(arr == 0))
    negatives = int(np.count_nonzero(arr < 0))
    if negatives:
        warnings.warn(f"{negatives} negative {name} values omitted from log plot", RuntimeWarning)
        arr = arr.copy()
        arr[arr < 0] = np.nan
    if zeros:
        warnings.warn(f"{zeros} zero {name} values shown at floor {floor:.1e}", RuntimeWarning)
    return np.where(arr == 0, floor, arr)


def display_name(mapping: Mapping[str, str], value: str) -> str:
    return mapping.get(value, value.replace("_", " ").title())


def scientific(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return "0" if value == 0 else f"{value:.{digits}e}"


def setup_axis(axis: plt.Axes, title: str, xlabel: str, ylabel: str,
               log_y: bool = False, y_limits: tuple[float, float] | None = None) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", alpha=.25)
    if log_y:
        axis.set_yscale("log")
        axis.grid(True, which="minor", alpha=.12)
    if y_limits:
        axis.set_ylim(*y_limits)


def style_for_series(basis: str, dtype: str) -> dict[str, object]:
    return {"marker": MARKERS.get(basis, "o"), "linestyle": LINESTYLES.get(dtype, "-"),
            "linewidth": 1.7, "markersize": 5.0}


def legend_label(basis: str, dtype: str) -> str:
    return f"{display_name(BASIS_DISPLAY, basis)} ({display_name(DTYPE_DISPLAY, dtype)})"


def add_legend_below(axis: plt.Axes, columns: int) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.18),
                    ncol=min(columns, len(handles)), frameon=False)


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(.5, .005, text, ha="center", va="bottom", fontsize=8)


def save_figure(fig: plt.Figure, context: PlotContext, stem: str) -> list[Path]:
    context.output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in context.formats:
        path = context.output_directory / f"{stem}.{ext}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": .04}
        if ext == "png":
            kwargs["dpi"] = context.dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
        print(f"Wrote {path}")
    plt.close(fig)
    return paths


def plot_metric_vs_degree(frame: pd.DataFrame, context: PlotContext, *, metric: str,
                          stem: str, title: str, ylabel: str, log_y: bool,
                          y_limits: tuple[float, float] | None = None) -> list[Path]:
    if metric not in frame.columns:
        print(f"Skipping {stem}: missing {metric}")
        return []
    selected = frame[(frame["dimension"] == context.detail_dimension) &
                     (frame["scenario"] == context.detail_scenario)].copy()
    selected = finite_metric_rows(selected, metric)
    if selected.empty:
        print(f"Skipping {stem}: no rows")
        return []
    grouped = aggregate_metric(selected, ["density_polynomial_degree", "basis", "dtype"], metric)
    fig, axis = plt.subplots(figsize=(context.figure_width, context.figure_height))
    for basis in ordered_present(grouped["basis"].unique(), BASIS_ORDER):
        for dtype in ordered_present(grouped["dtype"].unique(), DTYPE_ORDER):
            subset = grouped[(grouped["basis"] == basis) & (grouped["dtype"] == dtype)]
            subset = subset.sort_values("density_polynomial_degree")
            if subset.empty:
                continue
            x = subset["density_polynomial_degree"].to_numpy(float)
            median = subset["median"].to_numpy(float)
            q1 = subset["q1"].to_numpy(float)
            q3 = subset["q3"].to_numpy(float)
            if log_y:
                median = safe_positive(median, context.positive_floor, metric)
                q1 = safe_positive(q1, context.positive_floor, metric)
                q3 = safe_positive(q3, context.positive_floor, metric)
            line = axis.plot(x, median, label=legend_label(basis, dtype),
                             **style_for_series(basis, dtype))[0]
            axis.fill_between(x, q1, q3, alpha=.12, color=line.get_color(), linewidth=0)
    setup_axis(axis, title, "Density polynomial degree", ylabel, log_y, y_limits)
    axis.set_xticks(sorted(int(v) for v in grouped["density_polynomial_degree"].unique()))
    add_legend_below(axis, context.legend_columns)
    add_footer(fig, f"Dimension {context.detail_dimension} | "
                    f"{display_name(SCENARIO_DISPLAY, context.detail_scenario)} | median and IQR")
    fig.subplots_adjust(bottom=.26)
    return save_figure(fig, context, stem)


def plot_dynamic_schedule_error(frame: pd.DataFrame, context: PlotContext) -> list[Path]:
    metric = "relative_integration_error"
    selected = frame[(frame["scenario"] == "dynamic_box_with_obstacle") &
                     (frame["dimension"] == context.detail_dimension)].copy()
    selected = finite_metric_rows(selected, metric)
    if selected.empty:
        return []
    grouped = aggregate_metric(selected, ["schedule", "basis", "dtype"], metric)
    schedules = [s for s in ordered_present(grouped["schedule"].unique(), SCHEDULE_ORDER) if s != "static"]
    if not schedules:
        return []
    x = np.arange(len(schedules), dtype=float)
    fig, axis = plt.subplots(figsize=(context.figure_width, context.figure_height))
    for basis in ordered_present(grouped["basis"].unique(), BASIS_ORDER):
        for dtype in ordered_present(grouped["dtype"].unique(), DTYPE_ORDER):
            values = []
            for schedule in schedules:
                match = grouped[(grouped["schedule"] == schedule) &
                                (grouped["basis"] == basis) & (grouped["dtype"] == dtype)]
                values.append(float(match["median"].iloc[0]) if not match.empty else np.nan)
            axis.plot(x, safe_positive(values, context.positive_floor, metric),
                      label=legend_label(basis, dtype), **style_for_series(basis, dtype))
    axis.set_xticks(x)
    axis.set_xticklabels([display_name(SCHEDULE_DISPLAY, s) for s in schedules], rotation=20, ha="right")
    setup_axis(axis, "Dynamic constrained-domain integration error by schedule",
               "Obstacle schedule", METRIC_LABELS[metric], True)
    add_legend_below(axis, context.legend_columns)
    add_footer(fig, f"Dimension {context.detail_dimension} | all degrees | median aggregation")
    fig.subplots_adjust(bottom=.31)
    return save_figure(fig, context, "polytope_dynamic_schedule_error")


def plot_dynamic_trajectory_error(frame: pd.DataFrame, context: PlotContext) -> list[Path]:
    metric = "relative_integration_error"
    selected = frame[(frame["scenario"] == "dynamic_box_with_obstacle") &
                     (frame["dimension"] == context.detail_dimension) &
                     (frame["schedule"] == context.detail_schedule)].copy()
    selected = finite_metric_rows(selected, metric)
    if selected.empty:
        return []
    degree = int(selected["density_polynomial_degree"].max())
    selected = selected[selected["density_polynomial_degree"] == degree]
    grouped = aggregate_metric(selected, ["trajectory_step", "basis", "dtype"], metric)
    fig, axis = plt.subplots(figsize=(context.figure_width, context.figure_height))
    for basis in ordered_present(grouped["basis"].unique(), BASIS_ORDER):
        for dtype in ordered_present(grouped["dtype"].unique(), DTYPE_ORDER):
            subset = grouped[(grouped["basis"] == basis) & (grouped["dtype"] == dtype)]
            subset = subset.sort_values("trajectory_step")
            if subset.empty:
                continue
            x = subset["trajectory_step"].to_numpy(float)
            median = safe_positive(subset["median"].to_numpy(float), context.positive_floor, metric)
            q1 = safe_positive(subset["q1"].to_numpy(float), context.positive_floor, metric)
            q3 = safe_positive(subset["q3"].to_numpy(float), context.positive_floor, metric)
            line = axis.plot(x, median, label=legend_label(basis, dtype),
                             **style_for_series(basis, dtype))[0]
            axis.fill_between(x, q1, q3, alpha=.12, color=line.get_color(), linewidth=0)
    setup_axis(axis, "Dynamic integration error across the obstacle trajectory",
               "Trajectory step", METRIC_LABELS[metric], True)
    axis.set_xticks(sorted(int(v) for v in grouped["trajectory_step"].unique()))
    add_legend_below(axis, context.legend_columns)
    add_footer(fig, f"Dimension {context.detail_dimension} | "
                    f"{display_name(SCHEDULE_DISPLAY, context.detail_schedule)} | density degree {degree}")
    fig.subplots_adjust(bottom=.26)
    return save_figure(fig, context, "polytope_dynamic_trajectory_error")


def plot_error_summary(frame: pd.DataFrame, context: PlotContext) -> list[Path]:
    specs = [("relative_integration_error", "Direct integration"),
             ("recovered_integral_relative_error", "Recovered integral")]
    specs = [(m, l) for m, l in specs if m in frame.columns and not finite_metric_rows(frame, m).empty]
    if not specs:
        return []
    rows = []
    for metric, label in specs:
        grouped = aggregate_metric(frame, ["basis", "dtype"], metric)
        for record in grouped.to_dict("records"):
            rows.append({"metric": metric, "label": label, **record})
    summary = pd.DataFrame(rows)
    series = [(b, d) for b in ordered_present(summary["basis"].unique(), BASIS_ORDER)
              for d in ordered_present(summary["dtype"].unique(), DTYPE_ORDER)]
    x = np.arange(len(specs), dtype=float)
    width = min(.14, .78 / max(1, len(series)))
    fig, axis = plt.subplots(figsize=(context.figure_width, context.figure_height))
    for i, (basis, dtype) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        values = []
        for metric, _ in specs:
            match = summary[(summary["metric"] == metric) & (summary["basis"] == basis) &
                            (summary["dtype"] == dtype)]
            values.append(float(match["median"].iloc[0]) if not match.empty else np.nan)
        bars = axis.bar(x + offset, safe_positive(values, context.positive_floor, "summary error"),
                        width=width, label=legend_label(basis, dtype))
        for bar in bars:
            bar.set_hatch("" if dtype == "float32" else "//")
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in specs])
    setup_axis(axis, "Aggregate constrained-polytope numerical errors",
               "Error metric", "Median relative error", True)
    add_legend_below(axis, context.legend_columns)
    add_footer(fig, "All scenarios, dimensions, degrees, trials, schedules, and trajectory steps")
    fig.subplots_adjust(bottom=.27)
    return save_figure(fig, context, "polytope_error_summary")


def extremum_record(frame: pd.DataFrame, metric: str) -> dict[str, object] | None:
    usable = finite_metric_rows(frame, metric)
    if usable.empty:
        return None
    row = usable.loc[usable[metric].idxmax()]
    keys = ["scenario", "schedule", "trajectory_step", "dimension", "source_polynomial_degree",
            "density_polynomial_degree", "trial", "basis", "dtype", "obstacle_half_width",
            "relative_integration_error", "perturbation_sensitivity", "basis_condition_number",
            "numerical_rank_fraction", "coefficient_noise_amplification",
            "recovered_integral_relative_error", "total_evaluation_ms"]
    result: dict[str, object] = {"selected_metric": metric, "selected_metric_value": float(row[metric])}
    for key in keys:
        if key in row.index and not pd.isna(row[key]):
            value = row[key].item() if isinstance(row[key], np.generic) else row[key]
            result[key] = value
    return result


def print_configuration(frame: pd.DataFrame, path: Path, context: PlotContext) -> None:
    print("\nPolytope plotting configuration\n=================================")
    print(f"results path: {path}")
    print(f"output directory: {context.output_directory}")
    print(f"formats: {list(context.formats)}")
    print(f"dimensions: {sorted(frame['dimension'].dropna().astype(int).unique())}")
    print(f"source degrees: {sorted(frame['source_polynomial_degree'].dropna().astype(int).unique())}")
    print(f"density degrees: {sorted(frame['density_polynomial_degree'].dropna().astype(int).unique())}")
    print(f"scenarios: {ordered_present(frame['scenario'].unique(), SCENARIO_ORDER)}")
    print(f"schedules: {ordered_present(frame['schedule'].unique(), SCHEDULE_ORDER)}")
    print(f"bases: {ordered_present(frame['basis'].unique(), BASIS_ORDER)}")
    print(f"dtypes: {ordered_present(frame['dtype'].unique(), DTYPE_ORDER)}")
    print(f"records: {len(frame)}")
    print(f"detail dimension: {context.detail_dimension}")
    print(f"detail scenario: {context.detail_scenario}")
    print(f"detail schedule: {context.detail_schedule}\n")


def print_metric_table(frame: pd.DataFrame, metric: str, heading: str) -> None:
    if metric not in frame.columns:
        return
    grouped = aggregate_metric(frame, ["scenario", "basis", "dtype"], metric)
    if grouped.empty:
        return
    print(heading)
    print("=" * len(heading))
    display = grouped.copy()
    for column in ("median", "q1", "q3"):
        display[column] = display[column].map(scientific)
    print(display.to_string(index=False), "\n")


def json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for row in frame.to_dict("records"):
        safe = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                safe[key] = None
            elif pd.isna(value):
                safe[key] = None
            else:
                safe[key] = value
        records.append(safe)
    return records


def write_summary(frame: pd.DataFrame, context: PlotContext, path: Path,
                  written: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for metric in ["relative_integration_error", "perturbation_sensitivity",
                   "basis_condition_number", "numerical_rank_fraction",
                   "coefficient_noise_amplification", "recovered_integral_relative_error",
                   "total_evaluation_ms"]:
        if metric in frame.columns:
            metrics[metric] = json_safe_records(
                aggregate_metric(frame, ["scenario", "basis", "dtype"], metric)
            )
    payload = {
        "records": int(len(frame)),
        "finite_core_records": int(frame["is_finite_core"].sum()),
        "dimensions": sorted(int(v) for v in frame["dimension"].dropna().unique()),
        "source_polynomial_degrees": sorted(int(v) for v in frame["source_polynomial_degree"].dropna().unique()),
        "density_polynomial_degrees": sorted(int(v) for v in frame["density_polynomial_degree"].dropna().unique()),
        "scenarios": ordered_present(frame["scenario"].unique(), SCENARIO_ORDER),
        "schedules": ordered_present(frame["schedule"].unique(), SCHEDULE_ORDER),
        "bases": ordered_present(frame["basis"].unique(), BASIS_ORDER),
        "dtypes": ordered_present(frame["dtype"].unique(), DTYPE_ORDER),
        "detail_dimension": context.detail_dimension,
        "detail_scenario": context.detail_scenario,
        "detail_schedule": context.detail_schedule,
        "positive_plotting_floor": context.positive_floor,
        "figures": [str(p) for p in written],
        "median_metrics": metrics,
        "maximum_relative_integration_error": extremum_record(frame, "relative_integration_error"),
        "maximum_condition_number": extremum_record(frame, "basis_condition_number"),
        "maximum_coefficient_noise_amplification": extremum_record(frame, "coefficient_noise_amplification"),
        "maximum_recovered_integral_relative_error": extremum_record(frame, "recovered_integral_relative_error"),
        "maximum_runtime_ms": extremum_record(frame, "total_evaluation_ms"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    try:
        frame = apply_filters(load_results(args.results_path), args)
        context = PlotContext(
            output_directory=args.output_directory,
            formats=args.formats,
            dpi=args.dpi,
            positive_floor=args.positive_floor,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            detail_dimension=choose_detail_dimension(frame, args.detail_dimension),
            detail_scenario=choose_detail_value(frame, "scenario", args.detail_scenario,
                                                SCENARIO_ORDER, "scenario"),
            detail_schedule=choose_detail_value(frame, "schedule", args.detail_schedule,
                                                SCHEDULE_ORDER, "schedule"),
            legend_columns=args.legend_columns,
        )
        print_configuration(frame, args.results_path, context)
        if not args.quiet:
            print_metric_table(frame, "relative_integration_error",
                               "Median integration error by scenario, basis, and dtype")
            print_metric_table(frame, "basis_condition_number",
                               "Median condition number by scenario, basis, and dtype")
            print_metric_table(frame, "coefficient_noise_amplification",
                               "Median coefficient-noise amplification by scenario, basis, and dtype")
            print_metric_table(frame, "recovered_integral_relative_error",
                               "Median recovered-integral error by scenario, basis, and dtype")
            print_metric_table(frame, "total_evaluation_ms",
                               "Median runtime by scenario, basis, and dtype")
        written: list[Path] = []
        plots = [
            ("relative_integration_error", "polytope_integration_error_vs_degree",
             "Constrained-polytope integration error versus polynomial degree",
             METRIC_LABELS["relative_integration_error"], True, None),
            ("basis_condition_number", "polytope_condition_number_vs_degree",
             "Basis condition number versus polynomial degree",
             METRIC_LABELS["basis_condition_number"], True, None),
            ("numerical_rank_fraction", "polytope_rank_fraction_vs_degree",
             "Retained numerical rank versus polynomial degree",
             METRIC_LABELS["numerical_rank_fraction"], False, (0, 1.05)),
            ("coefficient_noise_amplification", "polytope_coefficient_noise_amplification_vs_degree",
             "Coefficient-noise amplification versus polynomial degree",
             METRIC_LABELS["coefficient_noise_amplification"], True, None),
            ("recovered_integral_relative_error", "polytope_recovered_integral_error_vs_degree",
             "Recovered-integral error versus polynomial degree",
             METRIC_LABELS["recovered_integral_relative_error"], True, None),
            ("perturbation_sensitivity", "polytope_perturbation_sensitivity_vs_degree",
             "Coefficient-perturbation sensitivity versus polynomial degree",
             METRIC_LABELS["perturbation_sensitivity"], True, None),
            ("total_evaluation_ms", "polytope_runtime_vs_degree",
             "Constrained-polytope runtime versus polynomial degree",
             METRIC_LABELS["total_evaluation_ms"], True, None),
        ]
        for metric, stem, title, ylabel, log_y, limits in plots:
            written.extend(plot_metric_vs_degree(frame, context, metric=metric, stem=stem,
                                                  title=title, ylabel=ylabel,
                                                  log_y=log_y, y_limits=limits))
        written.extend(plot_dynamic_schedule_error(frame, context))
        written.extend(plot_dynamic_trajectory_error(frame, context))
        written.extend(plot_error_summary(frame, context))
        write_summary(frame, context, args.summary_path, written)
        if not written:
            warnings.warn("No figures were produced", RuntimeWarning)
            return 1
        print(f"\nGenerated {len(written)} figure files.")
        return 0
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
