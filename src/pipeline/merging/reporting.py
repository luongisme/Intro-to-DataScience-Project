"""Build in-memory tables for presenting merge results."""

from typing import Dict

import pandas as pd

from src.pipeline.contracts.schema import MergeResult, TableSchema


def build_merge_report_tables(result: MergeResult) -> Dict[str, pd.DataFrame]:
    """Format a merge result as ordered tables for notebook display.

    Support results from merge_data, execute_join_plan and execute_join.
    Keep a stable set of table names and columns, including empty sections
    when no join occurred or a result has no recorded report. Never infer
    missing counts as zero, a business grain or successful validation.

    Show current output dimensions and dtypes alongside the declared schema.
    Other counts and validation statuses are recorded at merge time; this
    function does not revalidate data that the caller subsequently changed.
    Key flags identify components of one composite unique key, not separate
    uniqueness guarantees. Matching counts refer to input LEFT rows even
    when a join expands the output. A false row_count_checked flag means
    'Not applicable', while other false validation flags mean 'Failed'.

    Return fresh DataFrames containing scalar display values. Preserve step
    order, repeated table names, composite/differently named keys, selected
    columns and renames. Explicit plans can have no declared grain or key.
    Do not mutate the result, load/save files, print/display output, update
    documents, clean rows or change pandas display settings.

    Raises:
        TypeError: If result or its data/schema/report has an invalid type.
    """
    if not isinstance(result, MergeResult):
        raise TypeError("result must be a MergeResult.")
    if not isinstance(result.data, pd.DataFrame) or not isinstance(result.schema, TableSchema):
        raise TypeError("MergeResult requires DataFrame data and a TableSchema.")
    if not isinstance(result.report, dict):
        raise TypeError("MergeResult.report must be a dictionary.")

    report = result.report
    columns = {
        "overview": ["Field", "Value"],
        "sources": ["Table", "Rows", "Columns"],
        "joins": ["Step", "Left table", "Right table", "Left keys", "Right keys",
                  "How", "Cardinality", "Require match"],
        "row_counts": ["Step", "Left rows", "Right rows", "Output rows",
                       "Output columns", "Expected rows"],
        "key_checks": ["Step", "Metric", "Count"],
        "column_selection": ["Step", "Selected right columns", "Renames"],
        "validation": ["Scope", "Check", "Recorded status"],
        "schema": ["Column", "Expected dtype", "Actual dtype", "Unique key component",
                   "Required non-null"],
    }
    rows = {name: [] for name in columns}
    overview = {
        "Mode": report.get("mode"),
        "Source": report.get("source"),
        "Base table": report.get("base", report.get("left_table")),
        "Grain": report.get("grain"),
        "Requested tables": report.get("requested_tables"),
        "Bridge tables": report.get("bridge_tables"),
        "Resolved tables": report.get("resolved_tables", report.get("tables")),
        "Current rows": len(result.data),
        "Current columns": len(result.data.columns),
        "Declared unique key": ", ".join(result.schema.unique_key) or "Not declared",
    }
    for field, value in overview.items():
        if value is None:
            value = "Not declared" if field == "Grain" else "Not reported"
        elif isinstance(value, (list, tuple)):
            value = ", ".join(value) or "None"
        rows["overview"].append([field, value])
    for name, shape in report.get("input_shapes", {}).items():
        rows["sources"].append([name, shape.get("rows"), shape.get("columns")])

    single_join = "steps" not in report and "right_table" in report
    steps = [report] if single_join else report.get("steps", [])
    validations = [] if single_join else [("Merge", report.get("validation", {}))]
    for position, step in enumerate(steps, 1):
        rows["joins"].append([
            position, step.get("left_table"), step.get("right_table"),
            ", ".join(step.get("left_on", [])) or "Not reported",
            ", ".join(step.get("right_on", [])) or "Not reported",
            step.get("how"), step.get("cardinality"), step.get("require_match"),
        ])
        expected = step.get("expected_rows", "Not reported")
        rows["row_counts"].append([
            position, step.get("left_rows"), step.get("right_rows"),
            step.get("output_rows"), step.get("output_columns"),
            "Not applicable" if expected is None else expected,
        ])
        matching = step.get("matching", {})
        counts = {
            "Left duplicate key rows": step.get("left_duplicate_key_rows"),
            "Right duplicate key rows": step.get("right_duplicate_key_rows"),
            "Invalid left key rows": step.get("left_invalid_key_rows"),
            "Invalid right key rows": step.get("right_invalid_key_rows"),
            "Matched left rows": matching.get("matched_rows"),
            "Unmatched left rows": matching.get("unmatched_rows"),
            "Distinct unmatched keys": matching.get("unmatched_keys"),
        }
        rows["key_checks"].extend([position, metric, count] for metric, count in counts.items())
        selected = step.get("selected_right_columns")
        renamed = step.get("rename")
        rows["column_selection"].append([
            position,
            ", ".join(selected) if selected is not None else "Not reported",
            str(renamed) if renamed else ("None" if renamed == {} else "Not reported"),
        ])
        validations.append((f"Join {position}", step.get("validation", {})))
    for scope, checks in validations:
        for check, value in checks.items():
            if check == "row_count_checked" and value is False:
                status = "Not applicable"
            elif value is True:
                status = "Passed"
            elif value is False:
                status = "Failed"
            else:
                status = "Not reported"
            rows["validation"].append([scope, check.replace("_", " ").capitalize(), status])

    for column in result.schema.columns:
        rows["schema"].append([
            column, result.schema.dtypes.get(column, "Not declared"),
            str(result.data[column].dtype) if column in result.data.columns else "Missing",
            column in result.schema.unique_key, column in result.schema.non_nullable_columns,
        ])
    return {
        name: pd.DataFrame(rows[name], columns=names, dtype=object)
        for name, names in columns.items()
    }
