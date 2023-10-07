from contextlib import contextmanager
from typing import Callable
from unittest.mock import patch

import numpy as np
import pytest
from packaging.version import Version

from fastparquet.json import (
    JsonCodecError,
    _codec_cache,
    _codec_classes,
    _get_cached_codec,
    json_decoder,
    json_encoder,
)


@contextmanager
def _clear_cache():
    _codec_cache.clear()
    try:
        yield
    finally:
        _codec_cache.clear()


@pytest.mark.parametrize(
    "data,has_float64",
    [
        (None, False),
        ([1, 1, 2, 3, 5], False),
        ([1.23, -3.45], False),
        ([np.float64(0.12), np.float64(4.56)], True),
        ([[1, 2, 4], ["x", "y", "z"]], False),
        ({"k1": "value", "k2": "à/è", "k3": 3}, False),
        ({"k1": [1, 2, 3], "k2": [4.1, 5.2, 6.3]}, False),
    ],
)
@pytest.mark.parametrize(
    "encoder_module, encoder_class",
    list(_codec_classes.items()),
)
@pytest.mark.parametrize(
    "decoder_module, decoder_class",
    list(_codec_classes.items()),
)
def test_engine(encoder_module, encoder_class, decoder_module, decoder_class, data, has_float64):
    if encoder_module == "rapidjson" and has_float64 and Version(np.__version__).major >= 2:
        pytest.skip(reason="rapidjson cannot json dump np.float64 on numpy 2")

    pytest.importorskip(encoder_module)
    pytest.importorskip(decoder_module)

    encoder_obj = encoder_class()
    decoder_obj = decoder_class()

    dumped = encoder_obj.dumps(data)
    assert isinstance(dumped, bytes)

    loaded = decoder_obj.loads(dumped)
    assert loaded == data


@pytest.mark.parametrize(
    "module, impl_class",
    list(_codec_classes.items()),
)
def test__get_cached_codec(module, impl_class):
    pytest.importorskip(module)

    missing_modules = set(_codec_classes) - {module}
    with patch.dict("sys.modules", {mod: None for mod in missing_modules}):
        with _clear_cache():
            result = _get_cached_codec()
    assert isinstance(result, impl_class)


@pytest.mark.parametrize(
    "module, impl_class",
    list(_codec_classes.items()),
)
def test__get_cached_codec_without_any_available_codec(module, impl_class):
    # it should never happen in real cases unless the json implementation is removed
    pytest.importorskip(module)

    missing_modules = set(_codec_classes)
    with patch.dict("sys.modules", {mod: None for mod in missing_modules}):
        with _clear_cache():
            with pytest.raises(JsonCodecError, match="No available json codecs"):
                _get_cached_codec()


@pytest.mark.parametrize(
    "module, impl_class",
    list(_codec_classes.items()),
)
def test__get_cached_codec_with_env_variable(module, impl_class):
    pytest.importorskip(module)

    with patch.dict("os.environ", {"FASTPARQUET_JSON_CODEC": module}):
        with _clear_cache():
            result = _get_cached_codec()
    assert isinstance(result, impl_class)


def test__get_cached_codec_with_env_variable_and_invalid_codec():
    with patch.dict("os.environ", {"FASTPARQUET_JSON_CODEC": "invalid"}):
        with _clear_cache():
            with pytest.raises(JsonCodecError, match="Unsupported json codec 'invalid'"):
                _get_cached_codec()


def test__get_cached_codec_with_env_variable_and_unavailable_codec():
    with patch.dict("os.environ", {"FASTPARQUET_JSON_CODEC": "orjson"}):
        with patch.dict("sys.modules", {"orjson": None}):
            with _clear_cache():
                with pytest.raises(JsonCodecError, match="Unavailable json codec 'orjson'"):
                    _get_cached_codec()


def test_cache():
    with _clear_cache():
        assert _codec_cache.env is None
        assert _codec_cache.instance is None

        _get_cached_codec()
        instance_1 = _codec_cache.instance
        assert _codec_cache.env == ""
        assert _codec_cache.instance is not None

        _get_cached_codec()
        assert _codec_cache.env == ""
        assert _codec_cache.instance is instance_1

        _codec_cache.clear()
        assert _codec_cache.env is None
        assert _codec_cache.instance is None


def test_cache_with_env_variable():
    with _clear_cache(), patch.dict("os.environ") as environ:
        assert _codec_cache.env is None
        assert _codec_cache.instance is None

        _get_cached_codec()
        instance_orig = _codec_cache.instance
        assert _codec_cache.env == ""
        assert _codec_cache.instance is not None

        for module, impl_class in _codec_classes.items():
            environ["FASTPARQUET_JSON_CODEC"] = module
            try:
                _get_cached_codec()
            except JsonCodecError:
                # do not fail if the library isn't installed during tests
                pass
            else:
                assert _codec_cache.env == module
                assert _codec_cache.instance is not instance_orig
                assert isinstance(_codec_cache.instance, impl_class)

        del environ["FASTPARQUET_JSON_CODEC"]
        _get_cached_codec()
        assert _codec_cache.env == ""
        assert _codec_cache.instance is not instance_orig
        assert type(_codec_cache.instance) == type(instance_orig)

        _codec_cache.clear()
        assert _codec_cache.env is None
        assert _codec_cache.instance is None


def test_json_encoder():
    with _clear_cache():
        result = json_encoder()
    assert isinstance(result, Callable)


def test_json_decoder():
    with _clear_cache():
        result = json_decoder()
    assert isinstance(result, Callable)
