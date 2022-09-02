import logging
from abc import ABC, abstractmethod
from functools import lru_cache

logger = logging.getLogger("parquet")


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


@lru_cache(maxsize=None)
def _get_json_impl():
    """Return the first available json encoder/decoder implementation."""
    for engine_class in [OrjsonImpl, UjsonImpl, RapidjsonImpl]:
        try:
            return engine_class()
        except ImportError:
            pass
    # slower but always available
    return JsonImpl()


def json_encoder():
    """Return the first available json encoder function."""
    return _get_json_impl().dumps


def json_decoder():
    """Return the first available json decoder function."""
    return _get_json_impl().loads
