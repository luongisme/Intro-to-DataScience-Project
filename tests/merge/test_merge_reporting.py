"""Presentation regressions for recorded reports and current table state."""

from copy import deepcopy

import pandas as pd
import pytest

from src.pipeline.api import merge_data
from src.pipeline.contracts.schema import JoinSpec, MergeResult, TableSchema
from src.pipeline.merging.merger import execute_join, execute_join_plan
from src.pipeline.merging.reporting import build_merge_report_tables

TABLE_NAMES = ['overview', 'sources', 'joins', 'row_counts', 'key_checks',
               'column_selection', 'validation', 'schema']


@pytest.fixture
def report_inputs():
    left = pd.DataFrame({'id': pd.Series([1, 2], dtype='int64'), 'key': pd.Series(['A', 'A'], dtype='string')})
    right = pd.DataFrame({'key': pd.Series(['A', 'A'], dtype='string'), 'label': pd.Series(['one', 'two'], dtype='string')})
    ls = TableSchema(['id', 'key'], {'id': 'int64', 'key': 'string'}, ['id'])
    rs = TableSchema(['key', 'label'], {'key': 'string', 'label': 'string'})
    return left, right, ls, rs


@pytest.mark.parametrize('producer', ['single', 'plan', 'api', 'empty_plan', 'empty_report'])
def test_report_supports_all_result_producers_without_mutation(report_inputs, producer, capsys):
    left, right, ls, rs = report_inputs
    join = JoinSpec('right', on='key', validate='m:m')
    if producer == 'single':
        result = execute_join(left, right, ls, rs, join)
    elif producer == 'plan':
        result = execute_join_plan('left', [join], {'left': left, 'right': right}, {'left': ls, 'right': rs})
    elif producer == 'empty_report':
        result = MergeResult(left.copy(), ls)
    else:
        result = merge_data(base='left', joins=[] if producer == 'empty_plan' else [join],
                            source_tables={'left': left, 'right': right}, source_schemas={'left': ls, 'right': rs})
    originals = (result.data.copy(deep=True), deepcopy(result.schema), deepcopy(result.report))
    options = (pd.get_option('display.max_rows'), pd.get_option('display.max_columns'))
    tables = build_merge_report_tables(result)
    assert list(tables) == TABLE_NAMES
    assert all(isinstance(table, pd.DataFrame) for table in tables.values())
    assert all(all(dtype == object for dtype in table.dtypes) for table in tables.values())
    assert (pd.get_option('display.max_rows'), pd.get_option('display.max_columns')) == options
    assert capsys.readouterr().out == ''
    pd.testing.assert_frame_equal(result.data, originals[0])
    assert result.schema == originals[1] and result.report == originals[2]
    if producer in ('empty_plan', 'empty_report'):
        for name in ['joins', 'row_counts', 'key_checks', 'column_selection']:
            assert tables[name].empty and len(tables[name].columns) > 0
    else:
        assert tables['row_counts'].iloc[0]['Output rows'] == 4
        matching = tables['key_checks'].set_index('Metric')['Count']
        assert matching['Matched left rows'] == 2
        assert matching['Unmatched left rows'] == 0
    again = build_merge_report_tables(result)
    tables['overview'].iloc[0, 1] = 'local edit'
    assert again['overview'].iloc[0, 1] != 'local edit'
    assert result.report == originals[2]


def test_report_distinguishes_not_applicable_failed_and_missing(report_inputs):
    left, right, ls, rs = report_inputs
    result = execute_join(left, right, ls, rs, JoinSpec('right', on='key', validate='m:m'))
    result.report['validation']['cardinality'] = False
    tables = build_merge_report_tables(result)
    statuses = tables['validation'].set_index('Check')['Recorded status']
    assert statuses['Row count checked'] == 'Not applicable'
    assert statuses['Cardinality'] == 'Failed'
    assert statuses['Output schema'] == 'Passed'
    absent = build_merge_report_tables(MergeResult(left, ls))
    overview = absent['overview'].set_index('Field')['Value']
    assert overview['Grain'] == 'Not declared'
    assert overview['Mode'] == 'Not reported'
    assert absent['sources'].empty


def test_report_keeps_large_counts_exact_without_revalidating_mutated_data(report_inputs):
    left, right, ls, rs = report_inputs
    result = execute_join(left, right, ls, rs, JoinSpec('right', on='key', validate='m:m'))
    result.report['left_rows'] = 2**60 + 1
    result.report['matching']['matched_rows'] = 2**60 + 1
    result.data.drop(index=result.data.index[1:], inplace=True)
    result.data['id'] = result.data['id'].astype('float64')
    tables = build_merge_report_tables(result)
    assert tables['row_counts'].iloc[0]['Left rows'] == 2**60 + 1
    assert tables['row_counts'].iloc[0]['Output rows'] == 4
    assert tables['overview'].set_index('Field').loc['Current rows', 'Value'] == 1
    types = tables['schema'].set_index('Column')
    assert types.loc['id', 'Expected dtype'] == 'int64'
    assert types.loc['id', 'Actual dtype'] == 'float64'


@pytest.mark.parametrize('defect', ['result', 'data', 'schema', 'report'])
def test_report_rejects_invalid_container_types(defect):
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    schema = TableSchema(['id'], {'id': 'int64'})
    result = {'result': object(), 'data': MergeResult([], schema),
              'schema': MergeResult(frame, {}), 'report': MergeResult(frame, schema, [])}[defect]
    with pytest.raises(TypeError):
        build_merge_report_tables(result)
