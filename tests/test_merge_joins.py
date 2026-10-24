"""Join semantics checked against a row-pair reference independent of pandas merge."""

from collections import Counter
from dataclasses import replace
from itertools import product
from unittest.mock import Mock

import pandas as pd
import pytest

from src.pipeline.api import merge_data
from src.pipeline.contracts.schema import JoinSpec, TableSchema
from src.pipeline.merging.merger import execute_join, execute_join_plan

MISSING = object()
LEFT_SCHEMA = TableSchema(['lid', 'lk', 'amount', 'flag'],
    {'lid': 'int64', 'lk': 'string', 'amount': 'int64', 'flag': 'bool'}, ['lid'])
RIGHT_SCHEMA = TableSchema(['rid', 'rk', 'weight'],
    {'rid': 'int64', 'rk': 'string', 'weight': 'uint64'}, ['rid'])


def make_inputs(left_keys, right_keys):
    left = pd.DataFrame({'lid': pd.Series(range(len(left_keys)), dtype='int64'),
        'lk': pd.Series(left_keys, dtype='string'),
        'amount': pd.Series([2**53 + 1] * len(left_keys), dtype='int64'),
        'flag': pd.Series([False] * len(left_keys), dtype='bool')})
    right = pd.DataFrame({'rid': pd.Series(range(len(right_keys)), dtype='int64'),
        'rk': pd.Series(right_keys, dtype='string'),
        'weight': pd.Series([2**64 - 1] * len(right_keys), dtype='uint64')})
    return left, right


def normalized_rows(frame):
    return Counter(tuple(MISSING if pd.isna(value) else value for value in row)
                   for row in frame.itertuples(index=False, name=None))


def reference_pairs(left, right, how):
    """Match only valid original keys, keeping each unmatched side when required."""
    pairs, matched_left, matched_right = [], set(), set()
    for i, key in enumerate(left['lk']):
        matches = [j for j, other in enumerate(right['rk'])
                   if not pd.isna(key) and key.strip()
                   and not pd.isna(other) and other.strip() and key == other]
        for j in matches:
            pairs.append((i, j))
            matched_left.add(i)
            matched_right.add(j)
        if not matches and how in ('left', 'outer'):
            pairs.append((i, None))
    if how in ('right', 'outer'):
        pairs.extend((None, j) for j in range(len(right)) if j not in matched_right)
    rows = []
    for i, j in pairs:
        values = ([MISSING] * len(left.columns) if i is None else left.iloc[i].tolist())
        values += ([MISSING] * len(right.columns) if j is None else right.iloc[j].tolist())
        rows.append(tuple(MISSING if value is MISSING or pd.isna(value) else value for value in values))
    return Counter(rows), len(matched_left)


@pytest.mark.parametrize('how,cardinality,required', list(product(
    ['left', 'inner', 'right', 'outer'], ['1:1', '1:m', 'm:1', 'm:m'], [False, True])))
def test_join_modes_preserve_values_and_declared_guarantees(how, cardinality, required):
    left, right = make_inputs(['A', 'B'], ['A', 'B', 'C'] if required else ['A', 'C'])
    originals = (left.copy(deep=True), right.copy(deep=True))
    join = JoinSpec('right', left_on='lk', right_on='rk', how=how,
                    validate=cardinality, require_match=required)
    result = execute_join(left, right, LEFT_SCHEMA, RIGHT_SCHEMA, join)
    expected, matched = reference_pairs(left, right, how)
    assert normalized_rows(result.data) == expected
    assert result.report['matching']['matched_rows'] == matched
    assert result.report['matching']['unmatched_rows'] == len(left) - matched
    expected_count = None
    if cardinality in ('1:1', 'm:1') and (how == 'left' or how == 'inner' and required):
        expected_count = len(left)
    elif cardinality in ('1:1', '1:m') and (how == 'right' or how == 'outer' and required):
        expected_count = len(right)
    assert result.report['expected_rows'] == expected_count
    assert result.report['validation']['row_count_checked'] is (expected_count is not None)
    assert list(result.data.columns) == LEFT_SCHEMA.columns + RIGHT_SCHEMA.columns
    assert isinstance(result.data.index, pd.RangeIndex)
    assert not any(name.startswith('__merge_validity__') for name in result.data.columns)
    assert all(int(value) == 2**53 + 1 for value in result.data['amount'].dropna())
    assert all(int(value) == 2**64 - 1 for value in result.data['weight'].dropna())
    pd.testing.assert_frame_equal(left, originals[0])
    pd.testing.assert_frame_equal(right, originals[1])


@pytest.mark.parametrize('how', ['left', 'inner', 'right', 'outer'])
@pytest.mark.parametrize('left_keys,right_keys', [
    (['A', 'A', None, '', ' '], ['A', 'A', None, '', ' ']),
    ([], ['A', None]), (['A', None], []), ([], []),
])
def test_invalid_keys_never_match_and_empty_sides_follow_join_type(how, left_keys, right_keys):
    left, right = make_inputs(left_keys, right_keys)
    left.index = pd.Index([0] * len(left), name='lk')
    right.index = pd.Index([3] * len(right), name='rk')
    result = execute_join(left, right, LEFT_SCHEMA, RIGHT_SCHEMA,
        JoinSpec('right', left_on='lk', right_on='rk', how=how, validate='m:m', require_match=False))
    expected, matched = reference_pairs(left, right, how)
    assert normalized_rows(result.data) == expected
    assert result.report['matching']['matched_rows'] == matched
    assert result.report['left_invalid_key_rows'] == sum(pd.isna(x) or not x.strip() for x in left_keys)
    assert result.report['right_invalid_key_rows'] == sum(pd.isna(x) or not x.strip() for x in right_keys)


@pytest.mark.parametrize('cardinality,left_keys,right_keys,expected_rows', [
    ('m:1', ['A', 'A'], ['A'], 2), ('1:m', ['A'], ['A', 'A'], 2),
    ('m:m', ['A', 'A'], ['A', 'A'], 4),
])
def test_real_expansion_is_allowed_only_by_declared_cardinality(cardinality, left_keys, right_keys, expected_rows):
    left, right = make_inputs(left_keys, right_keys)
    result = execute_join(left, right, LEFT_SCHEMA, RIGHT_SCHEMA,
        JoinSpec('right', left_on='lk', right_on='rk', validate=cardinality))
    assert len(result.data) == expected_rows
    assert result.report['matching']['matched_rows'] == len(left)
    with pytest.raises(pd.errors.MergeError, match='cardinality'):
        execute_join(left, right, LEFT_SCHEMA, RIGHT_SCHEMA,
                     JoinSpec('right', left_on='lk', right_on='rk', validate='1:1'))


def test_strict_match_checks_foreign_keys_before_joining():
    left, right = make_inputs(['A', 'missing', None], ['A', None])
    with pytest.raises(ValueError, match='orphan'):
        execute_join(left, right, LEFT_SCHEMA, RIGHT_SCHEMA,
                     JoinSpec('right', left_on='lk', right_on='rk'))


def test_composite_different_keys_projection_and_rename():
    left = pd.DataFrame({'id': [1, 2], 'a': pd.Series(['A', 'A'], dtype='string'), 'b': [1, 2]})
    right = pd.DataFrame({'x': pd.Series(['A', 'A'], dtype='string'), 'y': [2, 1],
                          'label': pd.Series(['second', 'first'], dtype='string'), 'unused': [8, 9]})
    ls = TableSchema(['id', 'a', 'b'], {'id': 'int64', 'a': 'string', 'b': 'int64'}, ['id'])
    rs = TableSchema(['x', 'y', 'label', 'unused'],
        {'x': 'string', 'y': 'int64', 'label': 'string', 'unused': 'int64'}, ['x', 'y'])
    result = merge_data(base='left', joins=[JoinSpec('right', left_on=['a', 'b'],
        right_on=['x', 'y'], columns=['label'], rename={'label': 'description'})],
        source_tables={'left': left, 'right': right}, source_schemas={'left': ls, 'right': rs})
    assert list(result.data.columns) == ['id', 'a', 'b', 'x', 'y', 'description']
    assert result.data['description'].tolist() == ['first', 'second']
    assert result.report['grain'] is None and result.report['bridge_tables'] == []
    assert result.schema.unique_key == ['id']


def test_plan_preflights_later_collision_before_first_join(monkeypatch):
    left = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    right = pd.DataFrame({'id': pd.Series([1], dtype='int64'), 'label': pd.Series(['a'], dtype='string')})
    ls = TableSchema(['id'], {'id': 'int64'}, ['id'])
    rs = TableSchema(['id', 'label'], {'id': 'int64', 'label': 'string'}, ['id'])
    executor = Mock(side_effect=AssertionError('No join should run before schema preflight'))
    monkeypatch.setattr('src.pipeline.merging.merger.execute_join', executor)
    with pytest.raises(ValueError, match='step 2'):
        execute_join_plan('left', [JoinSpec('right', on='id'), JoinSpec('again', on='id')],
                          {'left': left, 'right': right, 'again': right},
                          {'left': ls, 'right': rs, 'again': rs})
    executor.assert_not_called()


def test_empty_plan_returns_independent_copies_and_ignores_unused_entries():
    left, _ = make_inputs(['A'], [])
    left.index = [42]
    result = execute_join_plan('left', [], {'left': left, 'unused': object()},
                                {'left': LEFT_SCHEMA, 'unused': object()})
    pd.testing.assert_frame_equal(result.data, left)
    assert result.data is not left and result.schema is not LEFT_SCHEMA
    assert result.schema.columns is not LEFT_SCHEMA.columns
    assert result.report['steps'] == []
    assert result.report['input_shapes'] == {'left': {'rows': 1, 'columns': 4}}
    result.data.loc[42, 'amount'] = 0
    assert left.loc[42, 'amount'] == 2**53 + 1


def test_internal_guard_name_cannot_overwrite_user_columns():
    left, right = make_inputs(['A'], ['A'])
    left['__merge_validity__'] = pd.Series(['keep'], dtype='string')
    schema = replace(LEFT_SCHEMA, columns=LEFT_SCHEMA.columns + ['__merge_validity__'],
                      dtypes={**LEFT_SCHEMA.dtypes, '__merge_validity__': 'string'})
    result = execute_join(left, right, schema, RIGHT_SCHEMA,
                          JoinSpec('right', left_on='lk', right_on='rk'))
    assert result.data['__merge_validity__'].tolist() == ['keep']

@pytest.mark.parametrize('how', ['left', 'inner', 'right', 'outer'])
@pytest.mark.parametrize('dtype,left_keys,right_keys', [
    ('UInt64', [2**64-1, pd.NA, 2**64-2], [2**64-1, pd.NA, 2**64-3]),
    ('uint64', [2**64-1, 2**64-2], [2**64-1, 2**64-3]),
    ('boolean', [True, pd.NA, False], [True, pd.NA]),
    ('float64', [1.25, float('nan'), 2.5], [1.25, float('nan'), 3.75]),
    ('datetime64[us]', [pd.Timestamp('2024-01-01'), pd.NaT],
                       [pd.Timestamp('2024-01-01'), pd.NaT, pd.Timestamp('2024-01-02')]),
    ('datetime64[us, UTC]', [pd.Timestamp('2024-01-01', tz='UTC'), pd.NaT],
                            [pd.Timestamp('2024-01-01', tz='UTC'), pd.NaT]),
])
def test_shared_typed_keys_keep_precision_and_never_match_nulls(how, dtype, left_keys, right_keys):
    left = pd.DataFrame({'lid': pd.Series(range(len(left_keys)), dtype='int64'),
                          'key': pd.Series(left_keys, dtype=dtype)})
    right = pd.DataFrame({'rid': pd.Series(range(len(right_keys)), dtype='int64'),
                           'key': pd.Series(right_keys, dtype=dtype)})
    ls = TableSchema(['lid', 'key'], {'lid': 'int64', 'key': dtype}, ['lid'])
    rs = TableSchema(['rid', 'key'], {'rid': 'int64', 'key': dtype}, ['rid'])
    result = execute_join(left, right, ls, rs,
        JoinSpec('right', on='key', how=how, validate='m:m', require_match=False))
    pairs, matched_right = [], set()
    for i, key in enumerate(left['key']):
        matches = [j for j, other in enumerate(right['key'])
                   if not pd.isna(key) and not pd.isna(other) and key == other]
        pairs.extend((i, j) for j in matches)
        matched_right.update(matches)
        if not matches and how in ('left', 'outer'):
            pairs.append((i, MISSING))
    if how in ('right', 'outer'):
        pairs.extend((MISSING, j) for j in range(len(right)) if j not in matched_right)
    assert normalized_rows(result.data[['lid', 'rid']]) == Counter(pairs)
    assert str(result.data['key'].dtype) == dtype
    for row in result.data.itertuples(index=False):
        expected = right['key'].iloc[int(row.rid)] if pd.isna(row.lid) else left['key'].iloc[int(row.lid)]
        assert pd.isna(row.key) if pd.isna(expected) else row.key == expected
