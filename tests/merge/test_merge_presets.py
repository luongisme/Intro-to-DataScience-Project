"""Preset, API and typed-loader regressions with independent reference joins."""

from dataclasses import replace
from itertools import combinations
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_loader import DATASETS, INTERIM_DATASETS, load_data
from src.paths import RAW_DIR
from src.pipeline.api import merge_data
from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA, JoinSpec, TableSchema, build_source_schema
from src.pipeline.io.csv_io import read_csv_dataset, write_csv_dataset
from src.pipeline.io.loader import load_table
from src.pipeline.io.parquet_io import read_parquet_dataset, write_parquet_dataset
from src.pipeline.merging.planner import MERGE_TABLE_SPECS, build_combination_plan, resolve_bridge_tables
from src.pipeline.merging.registry import build_dataset_registry

TABLES = ('order_items', 'orders', 'customers', 'products', 'sellers')
SELECTIONS = [selected for size in range(2, 6) for selected in combinations(TABLES, size)]
RIGHT_KEYS = {'orders': 'order_id', 'customers': 'customer_id',
              'products': 'product_id', 'sellers': 'seller_id'}


@pytest.fixture
def canonical_sources(synthetic_order_items, synthetic_orders, synthetic_customers,
                      synthetic_products, synthetic_sellers):
    frames = dict(zip(TABLES, [synthetic_order_items, synthetic_orders, synthetic_customers,
                              synthetic_products, synthetic_sellers]))
    schemas = {name: build_source_schema(MERGE_TABLE_SPECS[name]) for name in TABLES}
    typed = {name: frame.loc[:, schemas[name].columns].astype(schemas[name].dtypes)
             for name, frame in frames.items()}
    return typed, schemas


def expected_selection(selected):
    """Express the fixed five-table topology without calling production planning."""
    needed = set(selected)
    if not needed.issubset({'orders', 'customers'}):
        needed.add('order_items')
    if 'customers' in needed and 'order_items' in needed:
        needed.add('orders')
    resolved = [name for name in TABLES if name in needed]
    base = 'order_items' if 'order_items' in needed else 'orders'
    name = ('merged_order_items' if len(selected) == 5
            else 'merged_' + '__'.join(source for source in TABLES if source in selected))
    return resolved, base, name


def reference_join(selected, frames):
    """Use plain pandas joins as an independent oracle for valid canonical data."""
    resolved, base, _ = expected_selection(selected)
    expected = frames[base].copy(deep=True)
    for source in resolved:
        if source != base:
            expected = expected.merge(frames[source], on=RIGHT_KEYS[source], how='left',
                                      validate='m:1', sort=False)
    return expected


@pytest.mark.parametrize('selected', SELECTIONS, ids=lambda x: '+'.join(x))
def test_all_presets_match_reference_and_roundtrip(selected, canonical_sources, tmp_path):
    frames, schemas = canonical_sources
    originals = {name: frame.copy(deep=True) for name, frame in frames.items()}
    resolved, base, dataset_name = expected_selection(selected)
    result = merge_data(tables=list(reversed(selected)), source_tables=frames, source_schemas=schemas)
    expected = reference_join(selected, frames)
    pd.testing.assert_frame_equal(result.data, expected, check_exact=True)
    assert result.report['requested_tables'] == list(selected)
    assert result.report['resolved_tables'] == resolved
    assert result.report['bridge_tables'] == [name for name in resolved if name not in selected]
    assert result.report['base'] == base
    assert result.report['grain'] == ('order_item' if base == 'order_items' else 'order')
    assert result.report['mode'] == 'preset' and result.report['source'] == 'memory'
    assert result.schema.columns == list(expected.columns)
    assert result.schema.unique_key == (['order_id', 'order_item_id'] if base == 'order_items' else ['order_id'])
    required = set(result.schema.unique_key) | {RIGHT_KEYS[name] for name in resolved if name != base}
    assert result.schema.non_nullable_columns == [name for name in expected.columns if name in required]
    assert all(step['matching']['unmatched_rows'] == 0 for step in result.report['steps'])
    assert resolve_bridge_tables(list(selected)) == resolved
    plan = build_combination_plan(list(selected))
    assert plan['resolved_tables'] == resolved and plan['base'] == base
    assert result.schema == INTERIM_DATASETS[dataset_name]['schema']
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, originals[name], check_exact=True)
    csv_path = write_csv_dataset(result.data, tmp_path / 'case.csv', result.schema)
    pd.testing.assert_frame_equal(read_csv_dataset(csv_path, result.schema), expected, check_exact=True)
    pytest.importorskip('pyarrow')
    parquet_path = write_parquet_dataset(result.data, tmp_path / 'case.parquet', result.schema)
    pd.testing.assert_frame_equal(read_parquet_dataset(parquet_path, result.schema), expected, check_exact=True)


@pytest.fixture(scope='module')
def real_sources():
    missing = [RAW_DIR / DATASETS[name] for name in TABLES if not (RAW_DIR / DATASETS[name]).is_file()]
    if missing:
        pytest.skip('Real Olist raw files are unavailable: ' + ', '.join(str(p) for p in missing))
    frames, schemas = {}, {}
    for name, spec in MERGE_TABLE_SPECS.items():
        frame = pd.read_csv(RAW_DIR / DATASETS[name], usecols=spec.usecols, dtype=spec.dtypes)
        for column in spec.date_columns:
            frame[column] = pd.to_datetime(frame[column], errors='raise').astype('datetime64[us]')
        frames[name] = frame.loc[:, spec.usecols]
        schemas[name] = build_source_schema(spec)
    return frames, schemas


@pytest.mark.integration
@pytest.mark.parametrize('selected', SELECTIONS, ids=lambda x: '+'.join(x))
def test_real_presets_match_independent_reference(selected, real_sources):
    frames, schemas = real_sources
    result = merge_data(tables=selected, source_tables=frames, source_schemas=schemas)
    expected = reference_join(selected, frames)
    pd.testing.assert_frame_equal(result.data, expected, check_exact=True)
    _, base, _ = expected_selection(selected)
    assert len(result.data) == (112650 if base == 'order_items' else 99441)
    assert result.data['order_id'].nunique() == (98666 if base == 'order_items' else 99441)
    if len(selected) == 5:
        assert result.schema == FULL_MERGE_SCHEMA


@pytest.mark.parametrize('selected', [None, 'orders', {'orders', 'customers'}, [], ['orders'],
                                      ['orders', 'orders'], ['orders', 'unknown']])
def test_planner_rejects_invalid_selections(selected):
    with pytest.raises((TypeError, ValueError)):
        build_combination_plan(selected)


def test_registry_returns_independent_configurations():
    first, second = build_dataset_registry(), build_dataset_registry()
    assert len(first) == 26
    assert set(first) == {expected_selection(selected)[2] for selected in SELECTIONS}
    first['merged_order_items']['schema'].columns.clear()
    first['merged_order_items']['joins'][0].rename['local'] = 'local_copy'
    assert len(second['merged_order_items']['schema'].columns) == 25
    assert second['merged_order_items']['joins'][0].rename == {}
    assert len(FULL_MERGE_SCHEMA.columns) == 25


@pytest.mark.parametrize('kwargs', [{}, {'base': 'orders'}, {'joins': []},
    {'tables': ['orders', 'customers'], 'base': 'orders', 'joins': []},
    {'base': 'custom', 'joins': []}, {'base': 'orders', 'joins': 'customers'},
    {'base': 'orders', 'joins': [object()]},
])
def test_api_rejects_invalid_mode_before_loading(kwargs, monkeypatch):
    loader = Mock(side_effect=AssertionError('Invalid mode must not load raw data'))
    monkeypatch.setattr('src.pipeline.api.load_table', loader)
    with pytest.raises((TypeError, ValueError)):
        merge_data(**kwargs)
    loader.assert_not_called()


@pytest.mark.parametrize('defect', ['only_frames', 'missing_frame', 'missing_schema', 'noncanonical', 'bad_dtype'])
def test_memory_mode_never_fills_missing_sources_from_disk(defect, canonical_sources, monkeypatch):
    frames, schemas = canonical_sources
    loader = Mock(side_effect=AssertionError('Memory mode must not read raw data'))
    monkeypatch.setattr('src.pipeline.api.load_table', loader)
    kwargs = {'tables': ['orders', 'customers'], 'source_tables': frames, 'source_schemas': schemas}
    if defect == 'only_frames':
        kwargs.pop('source_schemas')
    elif defect == 'missing_frame':
        kwargs['source_tables'] = {k: v for k, v in frames.items() if k != 'customers'}
    elif defect == 'missing_schema':
        kwargs['source_schemas'] = {k: v for k, v in schemas.items() if k != 'customers'}
    elif defect == 'noncanonical':
        kwargs['source_schemas'] = {**schemas, 'customers': replace(schemas['customers'], unique_key=[])}
    else:
        kwargs['source_tables'] = {**frames, 'orders': frames['orders'].astype({'order_id': 'object'})}
    with pytest.raises((TypeError, ValueError)):
        merge_data(**kwargs)
    loader.assert_not_called()


def test_raw_mode_loads_only_required_sources_once(canonical_sources, monkeypatch):
    frames, _ = canonical_sources
    loader = Mock(side_effect=lambda spec: frames[spec.dataset_name])
    monkeypatch.setattr('src.pipeline.api.load_table', loader)
    result = merge_data(base='orders', joins=[
        JoinSpec('customers', on='customer_id', columns=['customer_state'], rename={'customer_state': 'first_state'}),
        JoinSpec('customers', on='customer_id', columns=['customer_state'], rename={'customer_state': 'second_state'}),
    ])
    assert [call.args[0].dataset_name for call in loader.call_args_list] == ['orders', 'customers']
    assert result.report['source'] == 'raw'
    assert result.data['first_state'].equals(result.data['second_state'])


def test_typed_loader_reads_zip_dates_and_preserves_uncleaned_rows(synthetic_pipeline_dirs):
    customers = load_table(MERGE_TABLE_SPECS['customers'])
    orders = load_table(MERGE_TABLE_SPECS['orders'])
    assert customers['customer_zip_code_prefix'].tolist() == ['01310', '04571']
    assert str(orders['order_approved_at'].dtype) == 'datetime64[us]'
    assert orders['order_status'].tolist() == ['delivered', 'shipped']
    assert orders['order_delivered_customer_date'].isna().sum() == 1
    assert len(orders) == 2


def test_registered_interim_schema_cannot_be_weakened(tmp_path, monkeypatch):
    monkeypatch.setattr('src.data_loader.INTERIM_DIR', tmp_path)
    with pytest.raises(ValueError, match='fixed registered schema'):
        load_data('merged_order_items', stage='interim', schema=replace(FULL_MERGE_SCHEMA, unique_key=[]))
    with pytest.raises(TypeError, match='Unsupported interim'):
        load_data('merged_order_items', stage='interim', usecols=['order_id'])
    with pytest.raises(ValueError, match='Supply schema'):
        load_data('custom', stage='interim')
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    schema = TableSchema(['id'], {'id': 'int64'}, ['id'])
    write_csv_dataset(frame, tmp_path / 'custom.csv', schema)
    pd.testing.assert_frame_equal(load_data('custom', stage='interim', schema=schema, file_format='csv'), frame)

@pytest.mark.parametrize('defect', ['duplicate_customer', 'bad_timestamp', 'bad_price', 'missing_column'])
def test_typed_loader_rejects_invalid_raw_sources(synthetic_pipeline_dirs, defect):
    name = 'customers' if defect == 'duplicate_customer' else 'order_items' if defect == 'bad_price' else 'orders'
    path = synthetic_pipeline_dirs['raw'] / DATASETS[name]
    frame = pd.read_csv(path, dtype='string')
    if defect == 'duplicate_customer':
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif defect == 'bad_timestamp':
        frame.loc[0, 'order_approved_at'] = 'invalid timestamp'
    elif defect == 'bad_price':
        frame.loc[0, 'price'] = 'not a price'
    else:
        frame = frame.drop(columns='customer_id')
    frame.to_csv(path, index=False, encoding='utf-8')
    with pytest.raises(ValueError):
        load_table(MERGE_TABLE_SPECS[name])
    assert not synthetic_pipeline_dirs['interim'].exists()
