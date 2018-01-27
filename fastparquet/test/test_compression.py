from fastparquet.compression import (compress_data, decompress_data,
        compressions, decompressions)

import pytest


@pytest.mark.parametrize('fmt', compressions)
def test_compress_decompress_roundtrip(fmt):
    data = b'123' * 1000
    compressed = compress_data(data, compression=fmt)
    if fmt.lower() == 'uncompressed':
        assert compressed is data
    else:
        assert len(compressed) < len(data)

    decompressed = decompress_data(compressed, algorithm=fmt)
    assert data == decompressed


def test_compress_decompress_roundtrip_args_gzip():
    data = b'123' * 1000
    compressed = compress_data(
        data,
        compression={
            "type": "gzip",
            "args": {
                "compresslevel": 5,
            }
        }
    )
    assert len(compressed) < len(data)

    decompressed = decompress_data(compressed, algorithm="gzip")
    assert data == decompressed

def test_compress_decompress_roundtrip_args_lz4():
    pytest.importorskip('lz4')
    data = b'123' * 1000
    compressed = compress_data(
        data,
        compression={
            "type": "lz4",
            "args": {
                "compression_level": 5,
                "content_checksum": True,
                "block_size": 0,
                "block_checksum": True,
                "block_linked": True,
                "store_size": True,
            }
        }
    )
    assert len(compressed) < len(data)

    decompressed = decompress_data(compressed, algorithm="lz4")
    assert data == decompressed

def test_errors():
    with pytest.raises(RuntimeError) as e:
        compress_data(b'123', compression='not-an-algorithm')

    assert 'not-an-algorithm' in str(e)
    assert 'gzip' in str(e).lower()


def test_not_installed():
    compressions.pop('BROTLI', None)
    with pytest.raises(RuntimeError) as e:
        compress_data(b'123', compression=4)
    assert 'brotli' in str(e.value).lower()
