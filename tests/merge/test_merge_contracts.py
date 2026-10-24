"""Contract regressions for typed merge inputs and outputs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.pipeline.contracts.normalizer import normalize_table_types
from src.pipeline.contracts.schema import (
    FULL_MERGE_SCHEMA, ORDER_ITEMS_SPEC, SOURCE_PRIMARY_KEYS,
    JoinSpec, TableLoadSpec, TableSchema, build_source_schema,
)
from src.pipeline.contracts.schema_derivation import derive_join_schema
from src.pipeline.contracts.validator import (
    validate_foreign_keys, validate_source_keys, validate_table_contract,
    validate_table_schema,
)


def test_source_schema_contracts_are_independent():
    schema = build_source_schema(ORDER_ITEMS_SPEC)
    assert schema.columns == [
        'order_id', 'order_item_id', 'product_id', 'seller_id',
        'shipping_limit_date', 'price', 'freight_value',
    ]
    assert schema.dtypes == {
        'order_id': 'string', 'order_item_id': 'int64', 'product_id': 'string',
        'seller_id': 'string', 'shipping_limit_date': 'datetime64[us]',
        'price': 'float64', 'freight_value': 'float64',
    }
    assert schema.unique_key == ['order_id', 'order_item_id']
    assert schema.non_nullable_columns == schema.unique_key
    schema.columns.append('local_only')
    schema.unique_key.clear()
    assert 'local_only' not in ORDER_ITEMS_SPEC.usecols
    assert SOURCE_PRIMARY_KEYS['order_items'] == ['order_id', 'order_item_id']
    assert build_source_schema(ORDER_ITEMS_SPEC).unique_key == ['order_id', 'order_item_id']
    assert FULL_MERGE_SCHEMA.unique_key == ['order_id', 'order_item_id']
    assert len(FULL_MERGE_SCHEMA.columns) == 25


@pytest.mark.parametrize('spec,keys', [
    (TableLoadSpec('custom', ['id', 'id'], {'id': 'string'}), ['id']),
    (TableLoadSpec('custom', ['id'], {}), ['id']),
    (TableLoadSpec('custom', ['id'], {'id': 'string', 'extra': 'string'}), ['id']),
    (TableLoadSpec('custom', ['id'], {'id': 'string'}, ['id']), ['id']),
    (TableLoadSpec('custom', ['id'], {'id': 'string'}), []),
    (TableLoadSpec('custom', ['id'], {'id': 'string'}), ['absent']),
    (TableLoadSpec('custom', ['id'], {'id': 'string'}), None),
])
def test_invalid_load_specs_are_rejected(spec, keys):
    with pytest.raises((TypeError, ValueError)):
        build_source_schema(spec, key_columns=keys)


def test_normalization_preserves_input_extra_columns_and_missing():
    schema = TableSchema(['zip', 'count', 'stamp'], {
        'zip': 'string', 'count': 'Int64', 'stamp': 'datetime64[us]',
    })
    frame = pd.DataFrame({
        'zip': pd.Series(['01310', None], dtype='object'),
        'count': pd.Series([9007199254740993, pd.NA], dtype='Int64'),
        'stamp': ['2024-01-01 12:34:56.123456', None], 'extra': ['x', 'y'],
    })
    frame.index = [7, 7]
    original = frame.copy(deep=True)
    result = normalize_table_types(frame, schema)
    assert result is not frame
    assert result['zip'].iloc[0] == '01310'
    assert result['count'].iloc[0] == 9007199254740993
    assert result['stamp'].dtype == np.dtype('datetime64[us]')
    assert list(result.columns) == list(frame.columns)
    assert list(result.index) == [7, 7]
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result.isna(), frame.isna())
    with pytest.raises(ValueError, match='column mismatch'):
        validate_table_schema(result, schema)


@pytest.mark.parametrize('values,dtype', [
    ([1310], 'string'), (['True'], 'boolean'), (['1.5'], 'int64'),
    (['-1'], 'uint64'), (['128'], 'int8'), ([0.1], 'float32'),
    ([1700000000], 'datetime64[us]'),
    (['2024-01-01 00:00:00.123456789'], 'datetime64[us]'),
    (['not-a-date'], 'datetime64[us]'), (['anything'], 'category'),
])
def test_normalization_rejects_lossy_or_ambiguous_values(values, dtype):
    frame = pd.DataFrame({'value': values})
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match='Cannot normalize'):
        normalize_table_types(frame, TableSchema(['value'], {'value': dtype}))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('defect', ['reordered', 'extra', 'missing', 'duplicate', 'dtype', 'unit'])
def test_exact_schema_detects_structural_drift(defect):
    schema = TableSchema(['id', 'stamp'], {'id': 'int64', 'stamp': 'datetime64[us]'})
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64'),
                          'stamp': pd.Series(['2024-01-01'], dtype='datetime64[us]')})
    if defect == 'reordered':
        frame = frame[['stamp', 'id']]
    elif defect == 'extra':
        frame['other'] = 0
    elif defect == 'missing':
        frame = frame.drop(columns='id')
    elif defect == 'duplicate':
        frame.columns = ['id', 'id']
    elif defect == 'dtype':
        frame['id'] = frame['id'].astype('Int64')
    else:
        frame['stamp'] = frame['stamp'].astype('datetime64[ns]')
    with pytest.raises(ValueError):
        validate_table_schema(frame, schema)


def test_composite_primary_key_is_not_individual_column_uniqueness():
    frame = pd.DataFrame({'order_id': ['O1', 'O1'], 'order_item_id': [1, 2]})
    validate_source_keys(frame, 'order_items')
    validate_source_keys(frame.iloc[:0], 'order_items')
    with pytest.raises(ValueError, match='not unique'):
        validate_source_keys(pd.concat([frame, frame.iloc[[0]]]), 'order_items')


@pytest.mark.parametrize('bad_key', [None, '', '  ', '\t\n'])
def test_invalid_primary_key_components_are_rejected(bad_key):
    frame = pd.DataFrame({'id': pd.Series(['valid', bad_key], dtype='string')})
    with pytest.raises(ValueError):
        validate_source_keys(frame, 'custom', key_columns=['id'])


def test_contract_without_key_does_not_infer_registered_primary_key():
    schema = TableSchema(['value'], {'value': 'string'}, non_nullable_columns=['value'])
    frame = pd.DataFrame({'value': pd.Series(['', ''], dtype='string')})
    validate_table_contract(frame, schema, table_name='order_items', expected_rows=2)
    validate_table_contract(frame.iloc[:0], schema, expected_rows=0)
    with pytest.raises(ValueError):
        validate_table_contract(frame, schema, expected_rows=3)
    with pytest.raises(ValueError):
        validate_table_contract(pd.DataFrame({'value': pd.Series([pd.NA], dtype='string')}), schema)


@pytest.mark.parametrize('count', [True, -1, 1.0, '1'])
def test_contract_rejects_invalid_expected_count(count):
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    with pytest.raises((TypeError, ValueError)):
        validate_table_contract(frame, TableSchema(['id'], {'id': 'int64'}), expected_rows=count)


def test_composite_foreign_keys_match_whole_tuples_and_ignore_attributes():
    left = pd.DataFrame({'a': ['A', 'B', 'A', None, ' '], 'b': [1, 2, 2, 1, 1]})
    right = pd.DataFrame({'x': ['A', 'B', None, ' '], 'y': [1, 2, 1, 1], 'attr': [None] * 4})
    join = JoinSpec('right', left_on=['a', 'b'], right_on=['x', 'y'], require_match=False)
    report = validate_foreign_keys(left, right, join)
    assert report == {'left_rows': 5, 'matched_rows': 2, 'unmatched_rows': 3,
                      'unmatched_keys': 3, 'invalid_key_rows': 2}
    with pytest.raises(ValueError, match='orphan'):
        validate_foreign_keys(left, right, replace(join, require_match=True))


@pytest.mark.parametrize('change', [
    {'how': 'cross'}, {'validate': 'auto'}, {'on': 'id', 'left_on': 'id', 'right_on': 'id'},
    {'columns': ['absent']}, {'columns': ['label', 'label']},
    {'rename': {'id': 'new_id'}}, {'rename': {'label': 'id'}}, {'require_match': 'yes'},
])
def test_schema_derivation_rejects_unsafe_join_declarations(change):
    left = TableSchema(['id'], {'id': 'int64'}, ['id'])
    right = TableSchema(['id', 'label'], {'id': 'int64', 'label': 'string'}, ['id'])
    with pytest.raises((TypeError, ValueError)):
        derive_join_schema(left, right, replace(JoinSpec('right', on='id'), **change))


def test_schema_derivation_preserves_projection_order_and_nullable_precision():
    left = TableSchema(['id', 'amount'], {'id': 'int64', 'amount': 'uint64'}, ['id'], ['amount'])
    right = TableSchema(['key', 'flag', 'label'], {'key': 'int64', 'flag': 'bool', 'label': 'string'}, ['key'])
    join = JoinSpec('right', left_on='id', right_on='key', how='outer',
                    columns=['label', 'flag'], rename={'label': 'right_label'}, require_match=False)
    result = derive_join_schema(left, right, join)
    assert result.columns == ['id', 'amount', 'key', 'flag', 'right_label']
    assert result.dtypes == {'id': 'Int64', 'amount': 'UInt64', 'key': 'Int64',
                             'flag': 'boolean', 'right_label': 'string'}
    assert result.unique_key == []
    assert result.non_nullable_columns == []
    with pytest.raises(ValueError, match='dtypes differ'):
        derive_join_schema(left, replace(right, dtypes={**right.dtypes, 'key': 'Int64'}), join)
