"""Notebook workflow contracts on small full-table inputs without dataset I/O."""

import ast
import json
import os
from pathlib import Path
from unittest.mock import call, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config

from src.paths import PROJECT_ROOT
from src.eda import workflow

pytestmark = pytest.mark.usefixtures("eda_plot_cleanup")
NOTEBOOK_PATH = Path(os.environ.get("EDA_NOTEBOOK_PATH", PROJECT_ROOT / "notebooks/01_data_loading.ipynb"))


def _read_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8-sig"))


def _source(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def test_notebook_compiles_without_automatic_exports_or_extra_data_sources():
    notebook = _read_notebook()
    forbidden = {"to_csv", "to_parquet", "to_excel", "to_pickle", "to_feather", "to_json", "to_markdown", "write_text", "write_bytes", "savefig", "read_csv", "read_parquet"}
    loads = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(_source(cell))
        compile(tree, str(NOTEBOOK_PATH), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden)
                if isinstance(node.func, ast.Name) and node.func.id == "load_data":
                    loads.append(node)
    workflow_tree = ast.parse(Path(workflow.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(workflow_tree):
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Attribute) and node.func.attr in forbidden)
    cell_ids = {cell.get("id", "") for cell in notebook["cells"]}
    assert not any(cell_id.startswith("eda-g-") for cell_id in cell_ids)
    assert len(loads) == 1
    assert ast.literal_eval(loads[0].args[0]) == "merged_order_items"
    assert {keyword.arg: ast.literal_eval(keyword.value) for keyword in loads[0].keywords}["stage"] == "interim"


@pytest.mark.parametrize("working_directory", ["root", "notebooks"])
@pytest.mark.parametrize("population", ["baseline", "issues", "empty"])
def test_notebook_runs_with_small_full_table_preserving_input_and_denominators(eda_full_frame, tmp_path, monkeypatch, working_directory, population):
    frame = eda_full_frame.copy(deep=True)
    if population == "issues":
        frame.loc[frame["order_id"].eq("O1"), "order_approved_at"] = pd.NaT
        frame.iloc[2, frame.columns.get_loc("order_delivered_customer_date")] = pd.Timestamp("2017-02-01 12:00")
        frame.loc[frame["product_id"].eq("P1"), "product_weight_g"] = 0.
        for column in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
            frame.loc[frame["product_id"].eq("P2"), column] = float("nan")
        frame.loc[frame["product_id"].eq("P2"), "product_category_name"] = pd.NA
        frame.loc[frame["seller_id"].eq("S1"), "seller_city"] = "city  a"
        frame.iloc[1, frame.columns.get_loc("shipping_limit_date")] = pd.Timestamp("2020-01-01")
    elif population == "empty":
        frame = frame.iloc[:0].copy()
    before = frame.copy(deep=True)
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance(config=Config({"HistoryManager": {"enabled": False, "hist_file": ":memory:"}}))
    cwd = PROJECT_ROOT if working_directory == "root" else PROJECT_ROOT / "notebooks"
    notebook = _read_notebook()
    try:
        with patch.object(Path, "cwd", return_value=cwd), patch("src.data_loader.load_data", side_effect=lambda *args, **kwargs: frame.copy(deep=True)) as reader, patch("IPython.display.display"), patch("src.eda.workflow.display"):
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                with capture_output():
                    result = shell.run_cell(_source(cell), store_history=False)
                assert result.error_before_exec is None, result.error_before_exec
                assert result.error_in_exec is None, result.error_in_exec
            assert reader.call_args_list == [call("merged_order_items", stage="interim")]
        ns = shell.user_ns
        pd.testing.assert_frame_equal(ns["df"], before)
        pd.testing.assert_frame_equal(frame, before)
        assert ns["PROJECT_ROOT"] == PROJECT_ROOT
        assert plt.get_fignums() == []
        timeline = ns["timeline_details"]
        delivered = ns["delivered_completeness"]
        products = ns["product_completeness"]
        assert len(ns["entity_views"]["orders"]) == (0 if population == "empty" else 2)
        assert len(ns["entity_views"]["products"]) == (0 if population == "empty" else 2)
        assert timeline["timeline_combinations"]["orders"].sum() == timeline["any_ordering_flag"].sum()
        assert delivered["delivered_issue_overlap"].to_numpy().sum() == len(delivered["delivered_orders"])
        if population == "issues":
            assert len(delivered["delivered_missing_records"]) == 1
            assert timeline["flags"]["customer_before_carrier"].sum() == 1
            assert len(products["physical_product_records"]) == 2
            assert products["physical_issues"]["any_missing_physical"].sum() == 1
            assert products["physical_issues"]["nonpositive_product_weight_g"].sum() == 1
            assert len(ns["seller_city_spacing"]["city_space_records"]) == 1
            assert len(ns["deadline_calendar"]["far_deadline_records"]) == 1
        elif population == "empty":
            assert not timeline["any_ordering_flag"].any()
            assert delivered["delivered_missing_records"].empty
            assert products["physical_product_records"].empty
            assert ns["deadline_calendar"]["far_deadline_records"].empty
        else:
            assert not timeline["any_ordering_flag"].any()
            assert products["zero_freight"].sum() == 1
    finally:
        shell.history_manager.end_session()
        InteractiveShell.clear_instance()
        plt.close("all")
