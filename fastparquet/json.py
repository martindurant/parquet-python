import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache

logger = logging.getLogger("parquet")


class JsonCodecError(Exception):
    """Exception raised when trying to load an invalid json codec."""


class BaseImpl(ABC):
    @abstractmethod
    def dumps(self, data):
        """Serialize ``obj`` to a JSON formatted bytes instance containing UTF-8 data."""

    @abstractmethod
    def loads(self, s):
        """Deserialize ``s`` (str, bytes or bytearray containing JSON) to a Python object."""


class OrjsonImpl(BaseImpl):
    def __init__(self):
        import orjson

        logger.debug("Using orjson encoder/decoder")
        self.api = orjson

    def dumps(self, data):
        return self.api.dumps(data, option=self.api.OPT_SERIALIZE_NUMPY)

    def loads(self, s):
        return self.api.loads(s)


class UjsonImpl(BaseImpl):
    def __init__(self):
        import ujson

        logger.debug("Using ujson encoder/decoder")
        self.api = ujson

    def dumps(self, data):
        return self.api.dumps(
            data,
            ensure_ascii=False,
            escape_forward_slashes=False,
        ).encode("utf-8")

    def loads(self, s):
        return self.api.loads(s)


class RapidjsonImpl(BaseImpl):
    def __init__(self):
        import rapidjson

        logger.debug("Using rapidjson encoder/decoder")
        self.api = rapidjson

    def dumps(self, data):
        return self.api.dumps(data, ensure_ascii=False).encode("utf-8")

    def loads(self, s):
        return self.api.loads(s)


class JsonImpl(BaseImpl):
    def __init__(self):
        import json

        logger.debug("Using json encoder/decoder")
        self.api = json

    def dumps(self, data):
        return self.api.dumps(data, separators=(",", ":")).encode("utf-8")

    def loads(self, s):
        return self.api.loads(s)


def _get_specific_codec(codec):
    try:
        return _codec_classes[codec]()
    except KeyError:
        raise JsonCodecError(
            f"Unsupported json codec {codec!r}. Please use one of {list(_codec_classes)}"
        ) from None
    except ImportError:
        raise JsonCodecError(
            f"Unavailable json codec {codec!r}. Please install the required library."
        ) from None


@lru_cache(maxsize=None)
def _get_cached_codec():
    """Return the requested or first available json encoder/decoder implementation."""
    codec = os.getenv("FASTPARQUET_JSON_CODEC")
    if codec:
        return _get_specific_codec(codec)
    for codec in _codec_classes:
        try:
            return _get_specific_codec(codec)
        except JsonCodecError:
            pass
    raise JsonCodecError("No available json codecs.")


def clear_cached_codec():
    """Clear the cached codec so that it can be selected again."""
    _get_cached_codec.cache_clear()


def json_encoder():
    """Return the first available json encoder function."""
    return _get_cached_codec().dumps


def json_decoder():
    """Return the first available json decoder function."""
    return _get_cached_codec().loads


# module_name -> implementation_class
_codec_classes = {
    "orjson": OrjsonImpl,
    "ujson": UjsonImpl,
    "rapidjson": RapidjsonImpl,
    "json": JsonImpl,  # it should be the last
}
