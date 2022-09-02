from typing import Callable
from unittest.mock import patch

import numpy as np
import pytest

from fastparquet import json


@pytest.mark.parametrize(
    "data",
    [
        None,
        [1, 1, 2, 3, 5],
        [1.23, -3.45],
        [np.float64(0.12), np.float64(4.56)],
        [[1, 2, 4], ["x", "y", "z"]],
        {"k1": "value", "k2": "à/è", "k3": 3},
        {"k1": [1, 2, 3], "k2": [4.1, 5.2, 6.3]},
    ],
)
@pytest.mark.parametrize(
    "encoder_module, encoder_class",
    [
        ("orjson", json.OrjsonImpl),
        ("ujson", json.UjsonImpl),
        ("rapidjson", json.RapidjsonImpl),
        ("json", json.JsonImpl),
    ],
)
@pytest.mark.parametrize(
    "decoder_module, decoder_class",
    [
        ("orjson", json.OrjsonImpl),
        ("ujson", json.UjsonImpl),
        ("rapidjson", json.RapidjsonImpl),
        ("json", json.JsonImpl),
    ],
)
def test_engine(encoder_module, encoder_class, decoder_module, decoder_class, data):
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
    [
        ("orjson", json.OrjsonImpl),
        ("ujson", json.UjsonImpl),
        ("rapidjson", json.RapidjsonImpl),
        ("json", json.JsonImpl),
    ],
)
def test__get_json_impl(module, impl_class):
    pytest.importorskip(module)

    json._get_json_impl.cache_clear()
    missing_modules = {"orjson", "ujson", "rapidjson"} - {module}
    with patch.dict("sys.modules", {mod: None for mod in missing_modules}):
        result = json._get_json_impl()
    assert isinstance(result, impl_class)


def test_json_encoder():
    json._get_json_impl.cache_clear()
    result = json.json_encoder()
    assert isinstance(result, Callable)


def test_json_decoder():
    json._get_json_impl.cache_clear()
    result = json.json_decoder()
    assert isinstance(result, Callable)
