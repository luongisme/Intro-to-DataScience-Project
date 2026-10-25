"""Notebook-facing EDA inspections for the registered full merged dataset.

Each function accepts in-memory frames or earlier profiling results, displays
its tables/figures, and returns the complete evidence named in its docstring.
Pass entity_views from build_entity_views(), column_roles from
get_column_roles(), and earlier result mappings from the same input. All full
frames follow the registered 25-column merge schema. Numerical checks
remain in profiling.py/checks.py; plot primitives remain in plots.py.

No function reads or writes files, generates documents, applies cleaning, or
changes source values. Figures are displayed and closed. Returned input/view
references remain inspection-only; empty schema-bearing inputs are supported.
"""

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.eda.checks import (
    profile_order_composition,
    profile_text_formats,
    profile_timestamp_ordering,
    summarize_condition_by_group,
)
from src.eda.plots import (
    plot_frequency,
    plot_missingness_heatmap,
    plot_numeric_boxplot,
    plot_numeric_histogram,
)
from src.eda.profiling import (
    limit_frequencies,
    profile_columns,
    profile_frequencies,
    profile_numeric,
    profile_missing_patterns,
    profile_paired_missingness,
    profile_temporal_frequencies,
    profile_timestamp_coverage,
)
from src.pipeline.contracts.schema import ZIP_COLUMNS
from src.pipeline.contracts.validator import validate_source_keys


def inspect_numeric_distributions(entity_views: dict, column_roles: pd.DataFrame) -> dict:
    """Display six numeric distributions at their source grain.

    Return numeric summary tables keyed by source name. Histograms and boxplots
    use the original finite values; the quantile window affects display only.

    Returns:
        numeric_profiles: numeric summary tables keyed by source name.
    """
    numeric_units = {
        "price": "BRL", "freight_value": "BRL", "product_weight_g": "g",
        "product_length_cm": "cm", "product_height_cm": "cm", "product_width_cm": "cm",
    }
    numeric_profiles = {}

    for view_name, view in entity_views.items():
        measurement_columns = column_roles.loc[
            column_roles["source_table"].eq(view_name)
            & column_roles["role"].eq("measurement"), "column",
        ].tolist()
        if not measurement_columns:
            continue
        summary = profile_numeric(view, measurement_columns)
        numeric_profiles[view_name] = summary
        indexed = summary.set_index("column")
        print(f"Numeric measurements | {view_name} | N={len(view):,}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("Counts (missing_pct uses all rows)")
            display(indexed[[
                "rows", "finite", "missing", "missing_pct", "non_finite_non_null", "zero", "negative",
            ]].T)
            print("Distribution of finite values")
            display(indexed[[
                "min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean", "std",
            ]].T)
            print("IQR diagnostics (outside_iqr_pct uses finite values)")
            display(indexed[[
                "iqr", "iqr_lower", "iqr_upper", "below_iqr", "above_iqr", "outside_iqr_pct",
            ]].T)

        for column in measurement_columns:
            stats = indexed.loc[column]
            lower, upper = stats["p01"], stats["p99"]
            display_range = (float(lower), float(upper)) if pd.notna(lower) and pd.notna(upper) and lower < upper else None
            figure, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
            label = f"{column} ({numeric_units[column]})"
            plot_numeric_histogram(view[column], ax=axes[0], xlabel=label, title="Full range")
            plot_numeric_histogram(
                view[column], ax=axes[1], xlabel=label, value_range=display_range,
                title="1st–99th percentile window" if display_range else "Full range (no distinct quantile bounds)",
            )
            plot_numeric_boxplot(view[column], ax=axes[2], xlabel=label, title="Full range; 1.5-IQR whiskers")
            figure.suptitle(f"{column} | {view_name}")
            display(figure)
            plt.close(figure)
    return numeric_profiles


def inspect_category_distributions(entity_views: dict, column_roles: pd.DataFrame) -> dict:
    """Display source-grain category and ZIP frequencies.

    Return full and displayed frequency tables, keyed by (source, column).
    Top-15 displays preserve Other, Blank and Missing counts.

    Returns:
        frequency_tables: all observed value frequencies keyed by (source, column).
        frequency_display_tables: top-value tables including Other/Blank/Missing.
    """
    frequency_tables = {}
    frequency_display_tables = {}

    for view_name, view in entity_views.items():
        category_columns = column_roles.loc[
            column_roles["source_table"].eq(view_name)
            & column_roles["role"].isin(["category", "zip_code"]), "column",
        ].tolist()
        for column in category_columns:
            full_table = profile_frequencies(view[column])
            display_table = limit_frequencies(full_table, top_n=15)
            frequency_tables[(view_name, column)] = full_table
            frequency_display_tables[(view_name, column)] = display_table
            print(f"{column} | {view_name} | N={len(view):,}; top 15 plus special rows")
            with pd.option_context("display.max_rows", None):
                display(display_table)
            figure, axes = plot_frequency(
                display_table, top_n=None, title=f"{column} | {view_name}",
            )
            display(figure)
            plt.close(figure)
    return {
        'frequency_tables': frequency_tables,
        'frequency_display_tables': frequency_display_tables,
    }


def inspect_temporal_distributions(entity_views: dict, column_roles: pd.DataFrame) -> dict:
    """Display timestamp coverage and stored-clock distributions.

    Return coverage tables plus full/displayed calendar frequencies. Order
    timestamps use order grain; shipping deadlines use item grain.

    Returns:
        timestamp_profiles: coverage/precision tables keyed by source.
        temporal_frequency_tables: all observed value frequencies keyed by (source, column).
        temporal_display_tables: calendar frequencies as displayed.
    """
    timestamp_profiles = {}
    temporal_frequency_tables = {}
    temporal_display_tables = {}

    for view_name, view in entity_views.items():
        timestamp_columns = column_roles.loc[
            column_roles["source_table"].eq(view_name)
            & column_roles["role"].eq("timestamp"), "column",
        ].tolist()
        if not timestamp_columns:
            continue
        coverage = profile_timestamp_coverage(view, timestamp_columns)
        timestamp_profiles[view_name] = coverage
        print(f"Timestamp coverage | {view_name} | N={len(view):,}")
        with pd.option_context("display.max_columns", None):
            display(coverage)

        for column in timestamp_columns:
            figure, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
            for frequency, axis in zip(["month", "date", "weekday", "hour"], axes.flat):
                full_table = profile_temporal_frequencies(view[column], frequency)
                display_table = limit_frequencies(full_table, top_n=15) if frequency == "date" else full_table
                temporal_frequency_tables[(view_name, column, frequency)] = full_table
                temporal_display_tables[(view_name, column, frequency)] = display_table
                print(f"{column} | {view_name} | {frequency} | N={len(view):,}")
                with pd.option_context("display.max_rows", None):
                    display(display_table)
                plot_frequency(
                    display_table, top_n=None, horizontal=False, ax=axis,
                    title="Top 15 calendar dates plus Other/Missing" if frequency == "date" else frequency.title(),
                )
            figure.suptitle(f"{column} | {view_name} | Stored clock")
            display(figure)
            plt.close(figure)
    return {
        'timestamp_profiles': timestamp_profiles,
        'temporal_frequency_tables': temporal_frequency_tables,
        'temporal_display_tables': temporal_display_tables,
    }


def inspect_missing_patterns(df: pd.DataFrame, entity_views: dict, column_profile: pd.DataFrame) -> dict:
    """Display full-table and source-grain missingness patterns.

    Return the diagnostic frames, exact/count patterns and column profiles. The
    full_items frame aliases the input for inspection; do not modify it.

    Returns:
        missingness_frames: inspection references for full items and source-grain views.
        missingness_profiles: exact patterns and missing-count distributions by view.
        native_column_profiles: column completeness tables by view.
    """
    missingness_frames = {
        "full_items": df,
        **{name: view for name, view in entity_views.items() if name != "order_items"},
    }
    missingness_profiles = {}
    native_column_profiles = {}

    for view_name, view in missingness_frames.items():
        profile = profile_missing_patterns(view)
        missingness_profiles[view_name] = profile
        completeness = column_profile if view_name == "full_items" else profile_columns(view)
        native_column_profiles[view_name] = completeness
        print(f"{view_name} | N={len(view):,} | {len(view.columns)} selected columns")
        with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
            print("Columns with missing values (all counts at this view's grain)")
            incomplete = completeness.loc[completeness["missing"].gt(0), [
                "column", "rows", "missing", "missing_pct", "blank", "blank_pct",
            ]]
            if incomplete.empty:
                print("No missing values at this grain.")
            else:
                display(incomplete)
            print("Observed missing-count bins")
            display(profile["row_distribution"].loc[profile["row_distribution"]["rows"].gt(0)])
            print("All observed exact patterns; () means complete within the selection")
            display(profile["patterns"])
    return {
        'missingness_frames': missingness_frames,
        'missingness_profiles': missingness_profiles,
        'native_column_profiles': native_column_profiles,
    }


def inspect_missing_pairs(missingness_frames: dict, native_column_profiles: dict) -> dict:
    """Display paired missingness with directional heatmaps.

    Return every pair and the conditional-percentage matrix for full items,
    orders and products. Displays select columns with observed missingness.

    Returns:
        missing_pair_profiles: pair-count tables and directional matrices keyed by view.
    """
    missing_pair_profiles = {}

    for view_name in ["full_items", "orders", "products"]:
        view = missingness_frames[view_name]
        paired = profile_paired_missingness(view)
        missing_pair_profiles[view_name] = paired
        active = native_column_profiles[view_name].loc[
            native_column_profiles[view_name]["missing"].gt(0), "column",
        ].tolist()
        displayed_pairs = paired["pairs"].loc[
            paired["pairs"]["left"].isin(active) & paired["pairs"]["right"].isin(active)
        ]
        print(f"{view_name} | N={len(view):,} | {len(paired['pairs'])} total pairs; {len(displayed_pairs)} displayed")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("Pair counts (four states sum to N)")
            display(displayed_pairs[[
                "left", "right", "rows", "both_missing", "left_only_missing", "right_only_missing", "both_present",
            ]])
            print("Conditional percentages with explicit missing-row denominators")
            display(displayed_pairs[[
                "left", "right", "left_missing", "right_missing",
                "right_missing_given_left_pct", "left_missing_given_right_pct",
            ]])
        figure, axes = plot_missingness_heatmap(
            paired["conditional_pct"].loc[active, active],
            title=f"{view_name} | Also missing among rows missing the conditioning column",
        )
        display(figure)
        plt.close(figure)
    return missing_pair_profiles


def inspect_conditional_missingness(column_roles: pd.DataFrame, df: pd.DataFrame, entity_views: dict) -> dict:
    """Display missingness within status, month and geography groups.

    Return selected column lists, a validated order context, full grouped
    profiles and their overview. Missing grouping keys remain included.

    Returns:
        order_timestamp_columns: five order-level timestamp names.
        core_timestamp_columns: purchase, approval, carrier and customer timestamp names.
        product_measurement_columns: four product physical-measurement names.
        order_context: key-validated order attributes with customer state/purchase month.
        condition_profiles: complete grouped tables keyed by (target, grain, grouping).
        condition_overview: population, affected and displayed-group counts.
    """
    order_timestamp_columns = column_roles.loc[
        column_roles["source_table"].eq("orders") & column_roles["role"].eq("timestamp"), "column",
    ].tolist()
    core_timestamp_columns = [
        column for column in order_timestamp_columns if column != "order_estimated_delivery_date"
    ]
    product_measurement_columns = column_roles.loc[
        column_roles["source_table"].eq("products") & column_roles["role"].eq("measurement"), "column",
    ].tolist()

    # Add only the observed customer state to a diagnostic order view from the full input.
    # Check its key before grouping; never select an arbitrary customer state per order.
    order_context = df.loc[:, [*entity_views["orders"].columns, "customer_state"]].drop_duplicates().copy()
    validate_source_keys(order_context, "orders")
    order_context = order_context.assign(
        purchase_month=order_context["order_purchase_timestamp"].dt.strftime("%Y-%m")
    )
    product_view = entity_views["products"]
    core_missing = order_context[core_timestamp_columns].isna().any(axis=1)
    product_measurements_missing = product_view[product_measurement_columns].isna().any(axis=1)
    item_measurements_missing = df[product_measurement_columns].isna().any(axis=1)
    item_category_missing = df["product_category_name"].isna()

    condition_specs = [
        (column, "orders", order_context, order_context[column].isna(), ["order_status"], None)
        for column in order_timestamp_columns
    ]
    condition_specs.extend([
        ("any_core_timestamp", "orders", order_context, core_missing, ["order_status"], None),
        ("any_core_timestamp", "orders", order_context, core_missing, ["purchase_month"], None),
        ("any_core_timestamp", "orders", order_context, core_missing, ["customer_state"], None),
        ("any_product_measurement", "products", product_view, product_measurements_missing, ["product_category_name"], 15),
        ("product_category_name", "items", df, item_category_missing, ["seller_id"], 15),
        ("product_category_name", "items", df, item_category_missing, ["customer_state"], None),
        ("product_category_name", "items", df, item_category_missing, ["seller_state"], None),
        ("any_product_measurement", "items", df, item_measurements_missing, ["seller_id"], 15),
        ("any_product_measurement", "items", df, item_measurements_missing, ["customer_state"], None),
    ])
    condition_profiles = {}
    condition_overview_rows = []

    for target, grain, frame, condition, groups, display_limit in condition_specs:
        table = summarize_condition_by_group(frame, condition, groups)
        key = (target, grain, tuple(groups))
        condition_profiles[key] = table
        if display_limit is None:
            shown = table.sort_values(groups, na_position="last")
        else:
            shown = table.sort_values("affected_rows", ascending=False, kind="stable").head(display_limit)
        print(f"Missing {target} | grain={grain} | grouped by {groups}")
        print(
            f"All-group N={int(table['rows'].sum()):,}; affected={int(table['affected_rows'].sum()):,}; "
            f"showing {len(shown)} of {len(table)} groups; omitted groups={len(table) - len(shown)}"
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(shown)
        condition_overview_rows.append([
            target, grain, tuple(groups), int(table["rows"].sum()),
            int(table["affected_rows"].sum()), len(table), len(shown),
        ])

    condition_overview = pd.DataFrame(condition_overview_rows, columns=[
        "Missing target", "Grain", "Grouping", "Rows", "Affected rows", "Groups", "Displayed groups",
    ])
    print("Conditional-profile overview")
    with pd.option_context("display.max_rows", None):
        display(condition_overview)
    return {
        'order_timestamp_columns': order_timestamp_columns,
        'core_timestamp_columns': core_timestamp_columns,
        'product_measurement_columns': product_measurement_columns,
        'order_context': order_context,
        'condition_profiles': condition_profiles,
        'condition_overview': condition_overview,
    }


def inspect_status_missingness(condition_profiles: dict, order_timestamp_columns: list[str]) -> pd.DataFrame:
    """Display timestamp missing percentages within each order status.

    Return the status-by-timestamp matrix. Each percentage uses all orders in
    that status, including orders with missing timestamps.

    Returns:
        status_missing_rates: status-by-timestamp percentage matrix.
    """
    status_missing_rates = pd.concat(
        {
            target: condition_profiles[(target, "orders", ("order_status",))].set_index("order_status")["affected_pct_all"]
            for target in [*order_timestamp_columns, "any_core_timestamp"]
        },
        axis=1,
    ).sort_index(na_position="last")
    status_missing_rates.index.name = "Order status (denominator: all orders in status)"
    status_missing_rates.columns.name = "Missing timestamp / core set"
    display(status_missing_rates)
    figure, axes = plot_missingness_heatmap(
        status_missing_rates, title="Missing timestamp percentage within each order status",
    )
    display(figure)
    plt.close(figure)
    return status_missing_rates


def inspect_text_formats(column_roles: pd.DataFrame, missingness_frames: dict) -> dict:
    """Display original text-format checks at item and entity grain.

    Return summaries, candidate variants and bounded original examples by view.
    Temporary comparison keys never replace source text.

    Returns:
        text_format_profiles: format summaries, candidate variants and bounded examples by view.
    """
    string_id_columns = column_roles.loc[
        column_roles["role"].eq("identifier") & column_roles["expected_dtype"].eq("string"), "column",
    ].tolist()
    text_columns = column_roles.loc[column_roles["expected_dtype"].eq("string"), "column"].tolist()
    text_shape_patterns = {
        **{column: r"[0-9a-f]{32}" for column in string_id_columns},
        **{column: r"[0-9]{5}" for column in ZIP_COLUMNS},
        "customer_state": r"[A-Z]{2}", "seller_state": r"[A-Z]{2}",
    }
    text_format_profiles = {}

    for view_name, view in missingness_frames.items():
        selected = [column for column in text_columns if column in view.columns]
        profile = profile_text_formats(
            view, selected,
            patterns={column: text_shape_patterns[column] for column in selected if column in text_shape_patterns},
            sample_size=5,
        )
        text_format_profiles[view_name] = profile
        print(f"Original text formats | {view_name} | N={len(view):,}")
        indexed = profile["summary"].set_index("column")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print("Value counts and text lengths (checks can overlap)")
            display(indexed[["rows", "missing", "string_rows", "non_string", "blank", "min_length", "max_length"]].T)
            print("Formatting diagnostics")
            display(indexed[[
                "edge_whitespace", "repeated_whitespace", "control_character",
                "replacement_character", "pattern_evaluated", "pattern_mismatch",
                "pattern_mismatch_pct", "candidate_variant_groups",
            ]].T)
            print("Regex declarations (no pattern means no regex evaluation)")
            display(profile["summary"][["column", "pattern"]])
        print(f"Candidate variant forms: showing {min(15, len(profile['variants']))} of {len(profile['variants'])}; full table stays in text_format_profiles")
        if not profile["variants"].empty:
            display(profile["variants"].head(15))
        if not profile["examples"].empty:
            print("Bounded examples; row_position refers to iloc within the named view")
            display(profile["examples"])
    return text_format_profiles


def inspect_order_timeline(entity_views: dict) -> dict:
    """Display timestamp ordering and status-specific comparisons.

    Return the original nullable conditions, summary/examples and grouped status
    profiles. Unknown comparisons remain unknown.

    Returns:
        timeline_profile: ordering summary, nullable conditions and original examples.
        timeline_by_status: complete status groups keyed by ordering check.
    """
    timeline_profile = profile_timestamp_ordering(entity_views["orders"], sample_size=5)
    with pd.option_context("display.max_columns", None):
        display(timeline_profile["summary"])
    print(
        f"Orders with any ordering flag: {int(timeline_profile['conditions'].fillna(False).any(axis=1).sum()):,} "
        f"of {len(entity_views['orders']):,}; flags overlap"
    )
    if not timeline_profile["examples"].empty:
        print("Original timestamp examples (at most five per check)")
        with pd.option_context("display.max_columns", None):
            display(timeline_profile["examples"])

    timeline_by_status = {}
    for check in timeline_profile["conditions"].columns:
        table = summarize_condition_by_group(
            entity_views["orders"], timeline_profile["conditions"][check], ["order_status"],
        )
        timeline_by_status[check] = table
        if table["affected_rows"].sum():
            print(f"{check} by status; unknown_rows means one or both timestamps are absent")
            display(table)
    return {
        'timeline_profile': timeline_profile,
        'timeline_by_status': timeline_by_status,
    }


def inspect_measurement_plausibility(numeric_profiles: dict, df: pd.DataFrame) -> dict:
    """Display measurement counts and item deadlines before purchase.

    Return measurement counts, the nullable deadline condition and status
    groups. No measurements or deadlines are corrected.

    Returns:
        measurement_review: finite, missing, non-finite, zero and negative counts.
        deadline_before_purchase: nullable item-level raw timestamp comparison.
        deadline_by_status: comparable/unassessable/affected item counts by status.
    """
    measurement_review = pd.concat(
        [table.assign(grain=name) for name, table in numeric_profiles.items()], ignore_index=True,
    )[["grain", "column", "rows", "finite", "missing", "non_finite_non_null", "zero", "negative"]]
    display(measurement_review)

    deadline_comparable = df["shipping_limit_date"].notna() & df["order_purchase_timestamp"].notna()
    deadline_before_purchase = df["shipping_limit_date"].lt(
        df["order_purchase_timestamp"]
    ).astype("boolean").where(deadline_comparable, pd.NA)
    deadline_by_status = summarize_condition_by_group(df, deadline_before_purchase, ["order_status"])
    print("Item deadline before purchase by status (evaluated denominator: comparable item rows)")
    display(deadline_by_status)
    if deadline_before_purchase.any():
        print("First five flagged item rows; full diagnostic remains in deadline_before_purchase")
        display(df.loc[deadline_before_purchase.fillna(False), [
            "order_id", "order_item_id", "seller_id", "order_purchase_timestamp", "shipping_limit_date",
        ]].head(5))
    return {
        'measurement_review': measurement_review,
        'deadline_before_purchase': deadline_before_purchase,
        'deadline_by_status': deadline_by_status,
    }


def inspect_order_structure(df: pd.DataFrame) -> dict:
    """Display item, product, seller and deadline counts per order.

    Return complete composition tables and per-metric frequency tables. Example
    limits affect display only; no representative seller or deadline is
    selected.

    Returns:
        order_composition: complete order summary, order records and order-seller ranges.
        composition_frequency_tables: all observed value frequencies keyed by (source, column).
    """
    order_composition = profile_order_composition(df)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(order_composition["summary"])
    order_diagnostics = order_composition["orders"]
    composition_frequency_tables = {}
    for metric in ["item_rows", "products", "sellers", "deadlines"]:
        table = profile_frequencies(order_diagnostics[metric])
        composition_frequency_tables[metric] = table
        print(f"{metric} per order | N={len(order_diagnostics):,}")
        with pd.option_context("display.max_rows", None):
            display(table)

    composition_example_masks = {
        "Multiple sellers": order_diagnostics["sellers"].gt(1),
        "Multiple deadlines": order_diagnostics["deadlines"].gt(1),
        "Mixed equal/different state pairs": order_diagnostics["items_with_equal_states"].gt(0) & order_diagnostics["items_with_different_states"].gt(0),
        "Incomplete item numbering": ~order_diagnostics["consecutive_item_numbers"],
    }
    for label, mask in composition_example_masks.items():
        print(f"{label}: {int(mask.sum()):,} orders; at most five examples")
        if mask.any():
            with pd.option_context("display.max_columns", None):
                display(order_diagnostics.loc[mask].head(5))

    multi_deadline_orders = order_diagnostics.loc[order_diagnostics["deadlines"].gt(1), "order_id"].head(5)
    print("Order-seller deadline ranges for the first five multi-deadline orders")
    with pd.option_context("display.max_columns", None):
        display(order_composition["seller_deadlines"].loc[
            order_composition["seller_deadlines"]["order_id"].isin(multi_deadline_orders)
        ])
    return {
        'order_composition': order_composition,
        'composition_frequency_tables': composition_frequency_tables,
    }


def inspect_timeline_details(entity_views: dict, timeline_profile: dict, order_context: pd.DataFrame) -> dict:
    """Display timestamp reversals by clock precision and purchase month.

    Return order/flag views, bounded examples, precision counts, overlapping
    combinations and monthly tables. Flag-frequency plots describe the flagged
    cohort, not monthly failure rates.

    Returns:
        orders: reference to the validated order view.
        flags: boolean masks for observed True ordering conditions only.
        any_ordering_flag: union of observed True ordering flags by order.
        precision_examples: up to three original records per check/calendar bucket.
        timeline_precision: comparable, unknown, same-day and earlier-day counts.
        timeline_combinations: observed flag combinations within the flagged cohort.
        timeline_month_profiles: monthly tables using comparable-order denominators.
        flagged_month_frequencies: purchase-month counts within each flagged cohort.
    """
    orders = entity_views["orders"]
    flags = timeline_profile["conditions"].fillna(False)
    any_ordering_flag = flags.any(axis=1)
    active_checks = timeline_profile["summary"].loc[timeline_profile["summary"]["flagged_orders"].gt(0), "check"].tolist()
    precision_rows = []
    precision_examples = {}

    for row in timeline_profile["summary"].to_dict(orient="records"):
        check = row["check"]
        flagged = flags[check]
        start_day = orders[row["start_column"]].dt.normalize()
        end_day = orders[row["end_column"]].dt.normalize()
        same_day = flagged & start_day.eq(end_day)
        earlier_day = flagged & end_day.lt(start_day)
        precision_rows.append([
            check, row["evaluated_orders"], row["unassessable_orders"], int(flagged.sum()),
            int(same_day.sum()), int(earlier_day.sum()),
        ])
        for label, mask in [("same_calendar_day", same_day), ("earlier_calendar_date", earlier_day)]:
            precision_examples[(check, label)] = orders.loc[mask, [
                "order_id", "order_status", row["start_column"], row["end_column"],
            ]].head(3)
    timeline_precision = pd.DataFrame(precision_rows, columns=[
        "Check", "Comparable orders", "Unassessable orders", "Flagged orders", "Same-day reversal", "Earlier-date reversal",
    ])
    display(timeline_precision)
    print(f"Unique orders with any observed ordering flag: {int(any_ordering_flag.sum()):,} / {len(orders):,}")
    timeline_combinations = (
        flags.loc[any_ordering_flag, active_checks].value_counts().rename("orders").reset_index()
        if active_checks else pd.DataFrame({"orders": pd.Series(dtype="int64")})
    )
    print("Observed flag combinations within the flagged-order cohort; False means no observed True for that check")
    display(timeline_combinations)

    # The validated context preserves the order view's IDs and index.
    pd.testing.assert_series_equal(order_context["order_id"], orders["order_id"])
    timeline_month_profiles = {}
    flagged_month_frequencies = {}
    for check in ["carrier_before_approval", "customer_before_carrier"]:
        table = summarize_condition_by_group(order_context, timeline_profile["conditions"][check], ["purchase_month"])
        timeline_month_profiles[check] = table
        print(f"{check} by purchase month; percentages use comparable orders within each month")
        with pd.option_context("display.max_rows", None):
            display(table.sort_values("purchase_month", na_position="last"))
        frequency = profile_frequencies(order_context.loc[flags[check], "purchase_month"]).sort_values("value", na_position="last")
        flagged_month_frequencies[check] = frequency
        print(f"Frequency among flagged orders; pct_all denominator={int(flags[check].sum()):,}")
        with pd.option_context("display.max_rows", None):
            display(frequency)
        figure, axes = plot_frequency(frequency, top_n=None, horizontal=False, title=f"{check}: purchase months among flagged orders")
        display(figure)
        plt.close(figure)
    for (check, label), examples in precision_examples.items():
        if not examples.empty:
            print(f"{check} | {label} | at most three original examples")
            display(examples)
    return {
        'orders': orders,
        'flags': flags,
        'any_ordering_flag': any_ordering_flag,
        'precision_examples': precision_examples,
        'timeline_precision': timeline_precision,
        'timeline_combinations': timeline_combinations,
        'timeline_month_profiles': timeline_month_profiles,
        'flagged_month_frequencies': flagged_month_frequencies,
    }


def inspect_delivered_completeness(orders: pd.DataFrame, core_timestamp_columns: list[str], any_ordering_flag: pd.Series, order_context: pd.DataFrame) -> dict:
    """Display delivered-order completeness and overlap with ordering flags.

    Return the delivered diagnostic view, missing-core mask, original incomplete
    records, exact patterns, overlap counts and monthly groups. The input is not
    filtered.

    Returns:
        delivered_orders: reference to the validated order view.
        core_missing: missing-core mask aligned to all orders.
        delivered_missing_records: all delivered orders with a missing core timestamp.
        delivered_missing_profile: complete delivered-core missingness patterns/counts.
        delivered_missing_patterns: noncomplete patterns with missing-cohort percentages.
        delivered_issue_overlap: 2x2 missing-core versus observed-ordering-flag counts.
        delivered_missing_by_month: missing-core counts within delivered purchase months.
    """
    delivered_mask = orders["order_status"].eq("delivered").fillna(False)
    core_missing = orders[core_timestamp_columns].isna().any(axis=1)
    delivered_orders = orders.loc[delivered_mask]
    delivered_missing_records = delivered_orders.loc[core_missing.loc[delivered_mask]].copy()
    delivered_missing_profile = profile_missing_patterns(delivered_orders, core_timestamp_columns)
    delivered_missing_patterns = delivered_missing_profile["patterns"].loc[
        delivered_missing_profile["patterns"]["missing_count"].gt(0)
    ].copy()
    delivered_missing_patterns["pct_among_missing_orders"] = (
        100 * delivered_missing_patterns["rows"] / len(delivered_missing_records)
        if len(delivered_missing_records) else float("nan")
    )
    print(f"Delivered orders={len(delivered_orders):,}; missing any core timestamp={len(delivered_missing_records):,}")
    with pd.option_context("display.max_colwidth", None, "display.max_columns", None):
        display(delivered_missing_patterns)
    delivered_issue_overlap = pd.crosstab(
        core_missing.loc[delivered_mask], any_ordering_flag.loc[delivered_mask],
    ).reindex(index=[False, True], columns=[False, True], fill_value=0)
    delivered_issue_overlap.index.name = "Any missing core timestamp"
    delivered_issue_overlap.columns.name = "Any observed ordering flag"
    display(delivered_issue_overlap)
    delivered_missing_by_month = summarize_condition_by_group(
        order_context.loc[delivered_mask], core_missing.loc[delivered_mask], ["purchase_month"],
    )
    print("Delivered core-timestamp missingness by purchase month (all delivered orders in each month)")
    with pd.option_context("display.max_rows", None):
        display(delivered_missing_by_month.sort_values("purchase_month", na_position="last"))
    for missing_columns in delivered_missing_patterns["missing_columns"]:
        mask = pd.Series(True, index=delivered_missing_records.index)
        for column in core_timestamp_columns:
            mask = mask & (delivered_missing_records[column].isna() if column in missing_columns else delivered_missing_records[column].notna())
        print(f"Missing pattern {missing_columns}: at most five original orders")
        with pd.option_context("display.max_columns", None):
            display(delivered_missing_records.loc[mask, ["order_id", "order_status", *core_timestamp_columns]].head(5))
    return {
        'delivered_orders': delivered_orders,
        'core_missing': core_missing,
        'delivered_missing_records': delivered_missing_records,
        'delivered_missing_profile': delivered_missing_profile,
        'delivered_missing_patterns': delivered_missing_patterns,
        'delivered_issue_overlap': delivered_issue_overlap,
        'delivered_missing_by_month': delivered_missing_by_month,
    }


def inspect_product_completeness(entity_views: dict, product_measurement_columns: list[str], df: pd.DataFrame, orders: pd.DataFrame) -> dict:
    """Display physical-measurement issues and their item/order exposure.

    Return physical-issue masks, product records/exposure, category co-
    missingness and zero-freight profiles. Product counts and repeated item
    exposure have separate denominators.

    Returns:
        physical_issues: product-grain missing/nonpositive condition masks.
        physical_issue_summary: affected product, item and order counts with denominators.
        physical_product_records: original affected products plus item/order/seller exposure.
        missing_physical_category: physical/category co-missingness at product and item grain.
        zero_freight: nullable item-level zero-freight condition.
        zero_freight_by_status: zero-freight counts and denominators by status.
        zero_freight_by_category: complete zero-freight profiles by category.
    """
    products = entity_views["products"]
    physical_issues = {"any_missing_physical": products[product_measurement_columns].isna().any(axis=1)}
    physical_issues.update({
        f"nonpositive_{column}": products[column].le(0).fillna(False)
        for column in product_measurement_columns
    })
    physical_summary_rows = []
    any_product_issue = pd.Series(False, index=products.index)
    for issue, mask in physical_issues.items():
        product_ids = products.loc[mask, "product_id"]
        exposed_items = df["product_id"].isin(product_ids)
        physical_summary_rows.append([
            issue, int(mask.sum()), len(products), int(exposed_items.sum()), len(df),
            int(df.loc[exposed_items, "order_id"].nunique()), len(orders),
        ])
        any_product_issue = any_product_issue | mask
    physical_issue_summary = pd.DataFrame(physical_summary_rows, columns=[
        "Issue", "Products", "All products", "Item rows", "All item rows", "Orders", "All orders",
    ])
    display(physical_issue_summary)
    product_exposure = df.groupby("product_id", sort=False, observed=True).agg(
        item_rows=("order_id", "size"), orders=("order_id", "nunique"), sellers=("seller_id", "nunique"),
    )
    physical_product_records = products.loc[any_product_issue].join(product_exposure, on="product_id", validate="1:1")
    print(f"Unique affected products: {len(physical_product_records)}; original measurements and exposure counts")
    with pd.option_context("display.max_columns", None):
        display(physical_product_records)
    missing_physical_category_rows = []
    for grain, frame in [("products", products), ("items", df)]:
        missing_physical = frame[product_measurement_columns].isna().any(axis=1)
        both = missing_physical & frame["product_category_name"].isna()
        denominator = int(missing_physical.sum())
        missing_physical_category_rows.append([
            grain, denominator, int(both.sum()), 100 * int(both.sum()) / denominator if denominator else float("nan"),
        ])
    missing_physical_category = pd.DataFrame(missing_physical_category_rows, columns=[
        "Grain", "Missing physical", "Also missing category", "Category missing pct among missing physical",
    ])
    display(missing_physical_category)
    zero_freight = df["freight_value"].eq(0).astype("boolean").where(df["freight_value"].notna(), pd.NA)
    zero_freight_by_status = summarize_condition_by_group(df, zero_freight, ["order_status"])
    zero_freight_by_category = summarize_condition_by_group(df, zero_freight, ["product_category_name"])
    print(f"Zero freight: {int(zero_freight.sum()):,} item rows in {df.loc[zero_freight, 'order_id'].nunique():,} orders")
    display(zero_freight_by_status)
    print(f"Top 15 of {len(zero_freight_by_category)} category groups by zero-freight count; full table retained")
    display(zero_freight_by_category.sort_values("affected_rows", ascending=False, kind="stable").head(15))
    return {
        'physical_issues': physical_issues,
        'physical_issue_summary': physical_issue_summary,
        'physical_product_records': physical_product_records,
        'missing_physical_category': missing_physical_category,
        'zero_freight': zero_freight,
        'zero_freight_by_status': zero_freight_by_status,
        'zero_freight_by_category': zero_freight_by_category,
    }


def inspect_numeric_tails(numeric_profiles: dict, df: pd.DataFrame, entity_views: dict) -> dict:
    """Display category composition and original records in numeric tails.

    Return full grouped IQR profiles, at most three records per extreme and a
    finite-value summary. Reuse existing 1.5-IQR limits without selecting a
    cleaning threshold.

    Returns:
        iqr_group_profiles: complete category profiles keyed by grain/measurement.
        numeric_tail_records: up to three original records per grain/measurement/extreme.
        iqr_summary: existing IQR limits, finite/unknown and outside-limit counts.
    """
    iqr_group_profiles = {}
    numeric_tail_records = {}
    iqr_summary_rows = []
    for view_name, summary in numeric_profiles.items():
        frame = df if view_name == "order_items" else entity_views[view_name]
        grain = "items" if view_name == "order_items" else view_name
        for row in summary.to_dict(orient="records"):
            column = row["column"]
            values = frame[column]
            finite = values.notna() & ~values.isin([float("inf"), float("-inf")])
            condition = (values.lt(row["iqr_lower"]) | values.gt(row["iqr_upper"])).astype("boolean").where(finite, pd.NA)
            grouped = summarize_condition_by_group(frame, condition, ["product_category_name"])
            iqr_group_profiles[(grain, column)] = grouped
            iqr_summary_rows.append([
                grain, column, row["iqr_lower"], row["iqr_upper"], int(finite.sum()),
                int((~finite).sum()), int(condition.sum()), row["outside_iqr_pct"],
            ])
            print(f"{column} | {grain} | top 15 of {len(grouped)} category groups; full table retained")
            display(grouped.sort_values("affected_rows", ascending=False, kind="stable").head(15))
            reference_columns = ["order_id", "order_item_id", "product_id", "seller_id"] if grain == "items" else ["product_id"]
            reference_columns += ["product_category_name", column]
            for tail, ascending in [("minimum", True), ("maximum", False)]:
                records = frame.loc[finite].sort_values(column, ascending=ascending, kind="stable").loc[:, reference_columns].head(3)
                numeric_tail_records[(grain, column, tail)] = records
                print(f"{column}: {tail}; at most three original records")
                display(records)
    iqr_summary = pd.DataFrame(iqr_summary_rows, columns=[
        "Grain", "Column", "IQR lower", "IQR upper", "Finite rows", "Unassessable rows", "Outside IQR", "Outside IQR pct of finite",
    ])
    with pd.option_context("display.max_columns", None):
        display(iqr_summary)
    return {
        'iqr_group_profiles': iqr_group_profiles,
        'numeric_tail_records': numeric_tail_records,
        'iqr_summary': iqr_summary,
    }


def inspect_deadline_calendar(orders: pd.DataFrame, df: pd.DataFrame, order_timestamp_columns: list[str]) -> dict:
    """Display deadlines beyond the latest observed purchase year.

    Return the nullable comparison, original records, grain-specific counts and
    status profiles. The observation year is descriptive, not an SLA or validity
    cutoff.

    Returns:
        deadline_after_purchase_year: nullable item-level deadline-year comparison.
        far_deadline_records: all original items beyond the latest purchase year.
        far_deadline_counts: item, order and order-seller cohort counts.
        far_deadlines_by_status: full deadline-year comparison profiles by status.
    """
    purchase_min = orders["order_purchase_timestamp"].min()
    purchase_max = orders["order_purchase_timestamp"].max()
    print(f"Observed purchase range: {purchase_min} to {purchase_max}")
    if pd.notna(purchase_max):
        deadline_after_purchase_year = df["shipping_limit_date"].dt.year.gt(purchase_max.year).astype("boolean").where(df["shipping_limit_date"].notna(), pd.NA)
    else:
        deadline_after_purchase_year = pd.Series(pd.NA, index=df.index, dtype="boolean")
    far_deadline_records = df.loc[deadline_after_purchase_year.fillna(False), [
        "order_id", "order_item_id", "seller_id", "order_status", "customer_state", "seller_state",
        *order_timestamp_columns, "shipping_limit_date",
    ]].copy()
    far_deadline_counts = pd.DataFrame([
        ["Items", len(far_deadline_records)],
        ["Orders", far_deadline_records["order_id"].nunique()],
        ["Order-seller groups", len(far_deadline_records[["order_id", "seller_id"]].drop_duplicates())],
    ], columns=["Unit", "Count"])
    display(far_deadline_counts)
    far_deadlines_by_status = summarize_condition_by_group(df, deadline_after_purchase_year, ["order_status"])
    display(far_deadlines_by_status)
    print(f"Showing at most 10 of {len(far_deadline_records)} original item records; full table retained")
    with pd.option_context("display.max_columns", None):
        display(far_deadline_records.head(10))
    return {
        'deadline_after_purchase_year': deadline_after_purchase_year,
        'far_deadline_records': far_deadline_records,
        'far_deadline_counts': far_deadline_counts,
        'far_deadlines_by_status': far_deadlines_by_status,
    }


def inspect_seller_city_spacing(entity_views: dict, df: pd.DataFrame, orders: pd.DataFrame) -> dict:
    """Display repeated seller-city spaces and original spelling frequencies.

    Return affected seller records, their item-exposure mask and original forms
    within state. Whitespace/case comparison keys exist only in a temporary
    copy.

    Returns:
        city_space_records: affected sellers with original and display/comparison strings.
        city_affected_items: item-level exposure mask for the affected sellers.
        city_form_counts: original city spellings/counts within affected state/comparison keys.
    """
    sellers = entity_views["sellers"]
    extra_city_space = sellers["seller_city"].str.contains(r"\s{2,}", regex=True, na=False)
    city_context = sellers.copy().assign(
        comparison_city=sellers["seller_city"].str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()
    )
    city_space_records = city_context.loc[extra_city_space].copy()
    city_space_records["city_with_visible_spaces"] = city_space_records["seller_city"].str.replace(" ", "[space]", regex=False)
    city_affected_items = df["seller_id"].isin(city_space_records["seller_id"])
    print(
        f"Repeated-whitespace seller cities: {len(city_space_records):,} / {len(sellers):,} sellers; "
        f"{int(city_affected_items.sum()):,} / {len(df):,} item rows; "
        f"{df.loc[city_affected_items, 'order_id'].nunique():,} / {len(orders):,} orders"
    )
    display(city_space_records)
    city_keys = pd.MultiIndex.from_frame(city_space_records[["seller_state", "comparison_city"]]).unique()
    same_candidate_city = pd.MultiIndex.from_frame(city_context[["seller_state", "comparison_city"]]).isin(city_keys)
    city_form_counts = city_context.loc[same_candidate_city].groupby(
        ["seller_state", "comparison_city", "seller_city"], sort=False, observed=True, dropna=False,
    ).agg(sellers=("seller_id", "nunique")).reset_index()
    city_form_counts["city_with_visible_spaces"] = city_form_counts["seller_city"].str.replace(" ", "[space]", regex=False)
    print("All original spelling counts for affected comparison keys within the same state")
    display(city_form_counts)
    return {
        'city_space_records': city_space_records,
        'city_affected_items': city_affected_items,
        'city_form_counts': city_form_counts,
    }


def inspect_structure_overlap(order_composition: dict, orders: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Display overlapping seller, deadline and state-pair multiplicity.

    Return the diagnostic context, flags, full combinations/status groups and
    bounded example-order items. State equality compares observed strings; no
    route feature is assigned.

    Returns:
        composition_context: complete per-order composition with status.
        composition_flags: boolean masks for observed True ordering conditions only.
        composition_combinations: every observed combination with all-order denominators.
        composition_by_status: complete status groups for each composition condition.
        composition_item_examples: original items for up to three example orders per condition.
    """
    composition_context = order_composition["orders"].merge(
        orders[["order_id", "order_status"]], on="order_id", how="left", validate="1:1",
    )
    composition_flags = pd.DataFrame({
        "multiple_sellers": composition_context["sellers"].gt(1),
        "multiple_deadlines": composition_context["deadlines"].gt(1),
        "mixed_state_pairs": composition_context["items_with_equal_states"].gt(0) & composition_context["items_with_different_states"].gt(0),
    })
    composition_combinations = composition_flags.value_counts().rename("orders").reset_index()
    composition_combinations["all_orders"] = len(composition_context)
    composition_combinations["pct_all_orders"] = (
        100 * composition_combinations["orders"] / len(composition_context)
        if len(composition_context) else float("nan")
    )
    display(composition_combinations)
    composition_by_status = {}
    for condition_name in composition_flags:
        grouped = summarize_condition_by_group(composition_context, composition_flags[condition_name], ["order_status"])
        composition_by_status[condition_name] = grouped
        print(f"{condition_name} by status; denominator is all orders in each status")
        display(grouped)
    composition_item_examples = {}
    for condition_name in ["multiple_deadlines", "mixed_state_pairs"]:
        example_orders = composition_context.loc[composition_flags[condition_name]].head(3)
        original_items = df.loc[df["order_id"].isin(example_orders["order_id"]), [
            "order_id", "order_item_id", "seller_id", "customer_state", "seller_state", "shipping_limit_date",
        ]]
        composition_item_examples[condition_name] = original_items
        print(f"{condition_name}: first {len(example_orders)} example orders; showing at most 20 of {len(original_items)} original item rows")
        display(example_orders[["order_id", "order_status", "item_rows", "sellers", "deadlines", "items_with_equal_states", "items_with_different_states"]])
        display(original_items.head(20))
    return {
        'composition_context': composition_context,
        'composition_flags': composition_flags,
        'composition_combinations': composition_combinations,
        'composition_by_status': composition_by_status,
        'composition_item_examples': composition_item_examples,
    }
