"""Reversible storage and failure isolation for merge outputs."""

import csv
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.pipeline.contracts.schema import TableSchema
from src.pipeline.io.atomic_io import write_file_atomically
from src.pipeline.io.csv_codec import decode_csv_strings, encode_csv_strings
from src.pipeline.io.csv_io import read_csv_dataset, write_csv_dataset
from src.pipeline.io.csv_types import restore_csv_column
from src.pipeline.io.dataset_io import read_dataset, save_dataset
from src.pipeline.io.parquet_io import read_parquet_dataset, write_parquet_dataset


@pytest.fixture
def storage_case():
    frame = pd.DataFrame({
        'id': pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype='int64'),
        'text': pd.Series([pd.NA, '', 'NA', 'NULL', r'\N', r'\\abc', '01310', 'Việt,"\r\n\x00'], dtype='string'),
        'signed': pd.Series([-(2**63), 2**63-1, 2**53+1, pd.NA, 0, -1, 1, 2], dtype='Int64'),
        'unsigned': pd.Series([2**64-1, 0, pd.NA, 2**53+1, 1, 2, 3, 4], dtype='UInt64'),
        'float': pd.Series([np.finfo('float64').max, np.nextafter(0.0, 1.0), -0.0,
                            float('inf'), float('-inf'), pd.NA, 0.1, 1.0], dtype='Float64'),
        'small_float': pd.Series([0.1, -0.0, pd.NA, 1, 2, 3, 4, 5], dtype='Float32'),
        'flag': pd.Series([True, False, pd.NA, True, False, True, False, True], dtype='boolean'),
        'stamp': pd.Series(['2024-01-01 12:34:56.123456789', None] * 4, dtype='datetime64[ns]'),
        'zone': pd.Series(pd.to_datetime(['2018-11-04T03:30:00Z', None] * 4, utc=True)
                          .tz_convert('America/Sao_Paulo')).astype('datetime64[us, America/Sao_Paulo]'),
    })
    frame.index = range(10, 18)
    frame.attrs = {'not_serializable': object()}
    schema = TableSchema(list(frame.columns), {name: str(dtype) for name, dtype in frame.dtypes.items()}, ['id'], ['id'])
    return frame, schema


@pytest.mark.parametrize('file_format', ['csv', 'parquet'])
@pytest.mark.parametrize('empty', [False, True])
def test_storage_roundtrip_preserves_extreme_values_and_missing(storage_case, tmp_path, file_format, empty):
    if file_format == 'parquet':
        pytest.importorskip('pyarrow')
    frame, schema = storage_case
    if empty:
        frame = frame.iloc[:0].copy()
    original = frame.copy(deep=True)
    writer, reader = ((write_csv_dataset, read_csv_dataset) if file_format == 'csv'
                      else (write_parquet_dataset, read_parquet_dataset))
    destination = tmp_path / 'exact-name.tmp'
    assert writer(frame, destination, schema) == destination
    restored = reader(destination, schema)
    expected = frame.reset_index(drop=True)
    expected.attrs = {}
    pd.testing.assert_frame_equal(restored, expected, check_exact=True)
    assert restored.attrs == {}
    if not empty:
        assert np.signbit(restored.loc[2, 'float'])
        assert np.signbit(restored.loc[1, 'small_float'])
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize('dtype', ['string[python]', 'str'])
def test_csv_codec_distinguishes_literal_missing_and_empty_text(dtype):
    values = pd.Series([None, '', 'NA', 'NULL', r'\N', r'\abc', r'\\abc', '01310', 'Việt'], dtype=dtype, name='value')
    original = values.copy(deep=True)
    encoded = encode_csv_strings(values)
    assert not encoded.isna().any()
    restored = decode_csv_strings(encoded)
    pd.testing.assert_series_equal(restored, original)
    pd.testing.assert_series_equal(values, original)


@pytest.mark.parametrize('value', ['\\bad', '\\', '\\Nextra'])
def test_csv_decoder_rejects_invalid_escapes(value):
    with pytest.raises(ValueError, match='Invalid CSV text encoding'):
        decode_csv_strings(pd.Series([value], dtype='string'))


@pytest.mark.parametrize('values', [pd.Series(['a'], dtype='object'), pd.Series([1]), ['a']])
def test_csv_codec_rejects_implicit_type_coercion(values):
    with pytest.raises(TypeError):
        encode_csv_strings(values)
    with pytest.raises(TypeError):
        decode_csv_strings(values)


@pytest.mark.parametrize('cell,dtype', [
    ('1.5', 'Int64'), ('9223372036854775808', 'int64'), ('-1', 'UInt64'),
    ('true', 'boolean'), ('1', 'bool'), ('', 'float64'), ('NaN', 'float64'),
    ('1e400', 'float64'), ('1e-400', 'float64'), ('0.1', 'float32'),
    ('not-a-date', 'datetime64[us]'), ('2024-01-01T00:00:00.123456789', 'datetime64[us]'),
    ('2024-01-01T00:00:00+00:00', 'datetime64[us]'), ('2024-01-01', 'datetime64[us, UTC]'),
])
def test_csv_restoration_rejects_invalid_or_lossy_cells(cell, dtype):
    with pytest.raises(ValueError, match='Cannot restore CSV column'):
        restore_csv_column(pd.Series([cell], dtype='string', name='value'), dtype)


@pytest.mark.parametrize('content', [
    b'', b'\xef\xbb\xbfid,text\r\n1,a\r\n', b'text,id\r\na,1\r\n',
    b'id,id\r\n1,a\r\n', b'id,text\r\n1\r\n', b'id,text\r\n1,a,extra\r\n',
    b'id,text\r\n\r\n', b'id,text\r\n1,"open', b'id,text\r\n1,\xff\r\n',
    b'id,text\r\n1,a\r\n1,b\r\n', b'id,text\r\n\\N,a\r\n',
])
def test_csv_reader_rejects_malformed_records_and_contracts(tmp_path, content):
    schema = TableSchema(['id', 'text'], {'id': 'int64', 'text': 'string'}, ['id'])
    path = tmp_path / 'broken.csv'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        read_csv_dataset(path, schema)
    assert path.read_bytes() == content


def test_csv_reader_handles_long_fields_and_restores_global_limit(tmp_path):
    schema = TableSchema(['id', 'text'], {'id': 'int64', 'text': 'string'}, ['id'])
    frame = pd.DataFrame({'id': [1], 'text': pd.Series(['x' * 200000], dtype='string')})
    path = write_csv_dataset(frame, tmp_path / 'long.csv', schema)
    previous = csv.field_size_limit()
    csv.field_size_limit(32)
    try:
        pd.testing.assert_frame_equal(read_csv_dataset(path, schema), frame)
        assert csv.field_size_limit() == 32
        path.write_text('id,text\n1,"unclosed', encoding='utf-8')
        with pytest.raises(ValueError):
            read_csv_dataset(path, schema)
        assert csv.field_size_limit() == 32
    finally:
        csv.field_size_limit(previous)


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('file_format', ['csv', 'parquet'])
def test_timestamp_units_roundtrip_exactly(tmp_path, unit, file_format):
    if file_format == 'parquet':
        pytest.importorskip('pyarrow')
    dtype = f'datetime64[{unit}]'
    frame = pd.DataFrame({'stamp': pd.Series(['2024-01-01 12:34:56', None], dtype=dtype)})
    schema = TableSchema(['stamp'], {'stamp': dtype})
    writer, reader = ((write_csv_dataset, read_csv_dataset) if file_format == 'csv'
                      else (write_parquet_dataset, read_parquet_dataset))
    path = writer(frame, tmp_path / 'timestamps', schema)
    pd.testing.assert_frame_equal(reader(path, schema), frame, check_exact=True)


def test_parquet_reader_rejects_incompatible_storage_and_precision(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    path = tmp_path / 'foreign.parquet'
    pq.write_table(pa.table({'value': ['1']}), path)
    with pytest.raises(ValueError, match='stored type'):
        read_parquet_dataset(path, TableSchema(['value'], {'value': 'int64'}))
    pq.write_table(pa.table({'value': [0.1]}), path)
    with pytest.raises(ValueError, match='Cannot normalize'):
        read_parquet_dataset(path, TableSchema(['value'], {'value': 'float32'}))
    pq.write_table(pa.table({'value': pa.array([1], type=pa.int32())}), path)
    restored = read_parquet_dataset(path, TableSchema(['value'], {'value': 'int64'}))
    pd.testing.assert_frame_equal(restored, pd.DataFrame({'value': pd.Series([1], dtype='int64')}))


@pytest.mark.parametrize('file_format', ['csv', 'parquet'])
def test_low_level_writer_validates_before_overwriting(storage_case, tmp_path, file_format):
    frame, schema = storage_case
    path = tmp_path / 'keep'
    path.write_bytes(b'original')
    writer = write_csv_dataset if file_format == 'csv' else write_parquet_dataset
    with pytest.raises(ValueError):
        writer(frame.drop(columns='text'), path, schema)
    assert path.read_bytes() == b'original'


def test_atomic_write_publishes_only_completed_file(tmp_path):
    destination = tmp_path / 'dataset.csv'
    destination.write_bytes(b'old')
    received = []
    def writer(temporary):
        assert temporary.parent == destination.parent
        assert temporary.name.startswith('.merge-') and temporary.suffix == '.tmp'
        assert destination.read_bytes() == b'old'
        temporary.write_bytes(b'complete')
        received.append(temporary)
        return temporary
    assert write_file_atomically(destination, writer) == destination
    assert destination.read_bytes() == b'complete'
    assert len(received) == 1 and not received[0].exists()
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize('failure', [RuntimeError('writer failed'), KeyboardInterrupt('cancelled')])
def test_atomic_failure_preserves_target_and_exception(tmp_path, failure):
    destination = tmp_path / 'keep'
    destination.write_bytes(b'old')
    def writer(temporary):
        temporary.write_bytes(b'partial')
        raise failure
    with pytest.raises(type(failure)) as caught:
        write_file_atomically(destination, writer)
    assert caught.value is failure
    assert destination.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize('defect', ['wrong_return', 'removed_file', 'replace_failure'])
def test_atomic_rejects_invalid_callback_and_replace_failure(tmp_path, monkeypatch, defect):
    destination = tmp_path / 'keep'
    destination.write_bytes(b'old')
    def writer(temporary):
        if defect == 'wrong_return':
            return str(temporary)
        if defect == 'removed_file':
            temporary.unlink()
        else:
            temporary.write_bytes(b'new')
        return temporary
    if defect == 'replace_failure':
        monkeypatch.setattr('src.pipeline.io.atomic_io.os.replace', Mock(side_effect=PermissionError('locked')))
    with pytest.raises((ValueError, PermissionError)):
        write_file_atomically(destination, writer)
    assert destination.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_cleanup_failure_keeps_original_error_with_note(tmp_path, monkeypatch):
    error = RuntimeError('original error')
    original_unlink = Path.unlink
    def blocked_unlink(path, *args, **kwargs):
        if path.name.startswith('.merge-'):
            raise PermissionError('cleanup locked')
        return original_unlink(path, *args, **kwargs)
    def writer(temporary):
        raise error
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, 'unlink', blocked_unlink)
        with pytest.raises(RuntimeError) as caught:
            write_file_atomically(tmp_path / 'target', writer)
    assert caught.value is error
    assert any('cleanup locked' in note for note in error.__notes__)
    for temporary in tmp_path.glob('.merge-*.tmp'):
        temporary.unlink()


@pytest.mark.parametrize('defect', ['bad_name', 'wrong_columns', 'duplicate_key', 'empty_schema'])
def test_save_rejects_invalid_input_before_io(tmp_path, monkeypatch, defect):
    import src.pipeline.io.dataset_io as module
    frame = pd.DataFrame({'id': pd.Series([1, 2], dtype='int64')})
    schema = TableSchema(['id'], {'id': 'int64'}, ['id'])
    name = 'good'
    if defect == 'bad_name':
        name = '../escape'
    elif defect == 'wrong_columns':
        frame = frame.rename(columns={'id': 'other'})
    elif defect == 'duplicate_key':
        frame['id'] = [1, 1]
    else:
        frame, schema = pd.DataFrame(), TableSchema([], {})
    writer = Mock(side_effect=AssertionError('Invalid inputs must not reach a writer'))
    monkeypatch.setattr(module, 'write_parquet_dataset', writer)
    monkeypatch.setattr(module, 'write_csv_dataset', writer)
    output_dir = tmp_path / 'absent'
    with pytest.raises(ValueError):
        save_dataset(frame, output_dir, schema, dataset_name=name)
    writer.assert_not_called()
    assert not output_dir.exists()


def test_csv_fallback_preserves_old_parquet_and_default_read_preference(tmp_path, monkeypatch, capsys):
    pytest.importorskip('pyarrow')
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    schema = TableSchema(['id'], {'id': 'int64'}, ['id'])
    write_parquet_dataset(frame, tmp_path / 'sample.parquet', schema)
    updated = pd.DataFrame({'id': pd.Series([2], dtype='int64')})
    monkeypatch.setattr('src.pipeline.io.dataset_io.write_parquet_dataset', Mock(side_effect=PermissionError('locked parquet')))
    assert save_dataset(updated, tmp_path, schema, dataset_name='sample') == tmp_path / 'sample.csv'
    output = capsys.readouterr().out
    assert 'PermissionError' in output and 'Trying CSV fallback' in output
    pd.testing.assert_frame_equal(read_dataset(tmp_path, schema, dataset_name='sample'), frame)
    pd.testing.assert_frame_equal(read_dataset(tmp_path, schema, dataset_name='sample', file_format='csv'), updated)
    assert not list(tmp_path.glob('.merge-*.tmp'))


def test_corrupt_parquet_falls_back_but_direct_csv_never_checks_parquet(tmp_path, monkeypatch, capsys):
    pytest.importorskip('pyarrow')
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    schema = TableSchema(['id'], {'id': 'int64'}, ['id'])
    write_csv_dataset(frame, tmp_path / 'sample.csv', schema)
    (tmp_path / 'sample.parquet').write_bytes(b'broken parquet')
    pd.testing.assert_frame_equal(read_dataset(tmp_path, schema, dataset_name='sample'), frame)
    assert 'Trying CSV fallback' in capsys.readouterr().out
    forbidden = Mock(side_effect=AssertionError('Explicit CSV must skip Parquet'))
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_parquet_dataset', forbidden)
    pd.testing.assert_frame_equal(read_dataset(tmp_path, schema, dataset_name='sample', file_format='csv'), frame)
    forbidden.assert_not_called()
    assert 'CSV selected' in capsys.readouterr().out


@pytest.mark.parametrize('operation', ['read', 'save'])
def test_both_format_failures_preserve_original_causes_and_files(tmp_path, monkeypatch, operation):
    import src.pipeline.io.dataset_io as module
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    schema = TableSchema(['id'], {'id': 'int64'}, ['id'])
    errors = [PermissionError('parquet locked'), OSError('csv full')]
    for file_format in ['parquet', 'csv']:
        (tmp_path / f'sample.{file_format}').write_bytes(b'old ' + file_format.encode())
    for file_format, error in zip(['parquet', 'csv'], errors):
        monkeypatch.setattr(module, f'{"write" if operation == "save" else "read"}_{file_format}_dataset', Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        if operation == 'save':
            save_dataset(frame, tmp_path, schema, dataset_name='sample')
        else:
            read_dataset(tmp_path, schema, dataset_name='sample')
    assert isinstance(caught.value.__cause__, ExceptionGroup)
    assert caught.value.__cause__.exceptions == tuple(errors)
    assert 'parquet locked' in str(caught.value) and 'csv full' in str(caught.value)
    for file_format in ['parquet', 'csv']:
        assert (tmp_path / f'sample.{file_format}').read_bytes() == b'old ' + file_format.encode()
    assert not list(tmp_path.glob('.merge-*.tmp'))


def test_direct_csv_failure_keeps_identity_and_path_note(tmp_path, monkeypatch):
    error = FileNotFoundError('missing CSV')
    csv_reader = Mock(side_effect=error)
    parquet_reader = Mock(side_effect=AssertionError('Parquet must not run'))
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_csv_dataset', csv_reader)
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_parquet_dataset', parquet_reader)
    with pytest.raises(FileNotFoundError) as caught:
        read_dataset(tmp_path, TableSchema(['id'], {'id': 'int64'}), dataset_name='sample', file_format='csv')
    assert caught.value is error
    assert any(str(tmp_path / 'sample.csv') in note for note in error.__notes__)
    parquet_reader.assert_not_called()


@pytest.mark.parametrize('options', [
    {'file_format': 'CSV'}, {'file_format': None}, {'dataset_name': '../escape'},
    {'schema': TableSchema(['id'], {})}, {'schema': TableSchema([], {})},
])
def test_invalid_read_declarations_do_not_trigger_fallback(tmp_path, monkeypatch, options):
    reader = Mock(side_effect=AssertionError('Invalid declarations must not access files'))
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_csv_dataset', reader)
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_parquet_dataset', reader)
    kwargs = {'schema': TableSchema(['id'], {'id': 'int64'}), 'dataset_name': 'sample', **options}
    with pytest.raises((TypeError, ValueError)):
        read_dataset(tmp_path, **kwargs)
    reader.assert_not_called()


def test_save_interruption_does_not_try_csv(tmp_path, monkeypatch):
    error = KeyboardInterrupt('cancelled')
    csv_writer = Mock(side_effect=AssertionError('Interruptions must not fallback'))
    monkeypatch.setattr('src.pipeline.io.dataset_io.write_parquet_dataset', Mock(side_effect=error))
    monkeypatch.setattr('src.pipeline.io.dataset_io.write_csv_dataset', csv_writer)
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    with pytest.raises(KeyboardInterrupt) as caught:
        save_dataset(frame, tmp_path, TableSchema(['id'], {'id': 'int64'}), dataset_name='sample')
    assert caught.value is error
    csv_writer.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_csv_workflow_without_parquet_engine_in_fresh_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = '''
import builtins
import sys
from pathlib import Path
sys.dont_write_bytecode = True
original_import = builtins.__import__
def without_pyarrow(name, *args, **kwargs):
    if name == "pyarrow" or name.startswith("pyarrow."):
        raise ImportError("Simulated missing PyArrow engine")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_pyarrow
sys.path.insert(0, sys.argv[1])
import pandas as pd
from src.pipeline.contracts.schema import TableSchema
from src.pipeline.io.dataset_io import save_dataset, read_dataset
frame = pd.DataFrame({"id": pd.Series([1], dtype="int64"), "zip": pd.Series(["01310"], dtype="string")})
schema = TableSchema(["id", "zip"], {"id": "int64", "zip": "string"}, ["id"])
path = save_dataset(frame, Path(sys.argv[2]), schema, dataset_name="without_engine")
assert path.suffix == ".csv"
pd.testing.assert_frame_equal(read_dataset(Path(sys.argv[2]), schema, dataset_name="without_engine"), frame, check_exact=True)
assert not list(Path(sys.argv[2]).glob(".merge-*.tmp"))
print("Missing-engine CSV roundtrip passed")
'''
    completed = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path)],
                                capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'pip install' in completed.stdout and 'pyarrow' in completed.stdout
    assert 'Missing-engine CSV roundtrip passed' in completed.stdout


def test_directory_setup_error_does_not_try_another_format(tmp_path, monkeypatch):
    writer = Mock(side_effect=AssertionError('Directory setup must precede format attempts'))
    monkeypatch.setattr('src.pipeline.io.dataset_io.write_parquet_dataset', writer)
    monkeypatch.setattr('src.pipeline.io.dataset_io.write_csv_dataset', writer)
    blocked = tmp_path / 'not_a_directory'
    blocked.write_bytes(b'keep')
    frame = pd.DataFrame({'id': pd.Series([1], dtype='int64')})
    with pytest.raises(OSError):
        save_dataset(frame, blocked, TableSchema(['id'], {'id': 'int64'}), dataset_name='sample')
    writer.assert_not_called()
    assert blocked.read_bytes() == b'keep'


def test_read_interruption_does_not_try_csv(tmp_path, monkeypatch):
    error = KeyboardInterrupt('cancelled read')
    reader = Mock(side_effect=AssertionError('Interruptions must not fallback'))
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_parquet_dataset', Mock(side_effect=error))
    monkeypatch.setattr('src.pipeline.io.dataset_io.read_csv_dataset', reader)
    with pytest.raises(KeyboardInterrupt) as caught:
        read_dataset(tmp_path, TableSchema(['id'], {'id': 'int64'}), dataset_name='sample')
    assert caught.value is error
    reader.assert_not_called()
