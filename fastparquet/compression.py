
import gzip
from .thrift_structures import parquet_thrift
from .util import PY2

# TODO: use stream/direct-to-buffer conversions instead of memcopy

compressions = {
    'UNCOMPRESSED': lambda x: x
}
decompressions = {
    'UNCOMPRESSED': lambda x: x
}

# Gzip is present regardless
COMPRESSION_LEVEL = 9
if PY2:
    def gzip_compress_v2(data, compresslevel=COMPRESSION_LEVEL):
        from io import BytesIO
        bio = BytesIO()
        f = gzip.GzipFile(mode='wb',
                          compresslevel=compresslevel,
                          fileobj=bio)
        f.write(data)
        f.close()
        return bio.getvalue()
    def gzip_decompress_v2(data):
        import zlib
        return zlib.decompress(data,
                               16+15)
    compressions['GZIP'] = gzip_compress_v2
    decompressions['GZIP'] = gzip_decompress_v2
else:
    def gzip_compress_v3(data, compresslevel=COMPRESSION_LEVEL):
        return gzip.compress(data, compresslevel=compresslevel)
    compressions['GZIP'] = gzip_compress_v3
    decompressions['GZIP'] = gzip.decompress

try:
    import snappy
    compressions['SNAPPY'] = snappy.compress
    decompressions['SNAPPY'] = snappy.decompress
except ImportError:
    pass
try:
    import lzo
    compressions['LZO'] = lzo.compress
    decompressions['LZO'] = lzo.decompress
except ImportError:
    pass
try:
    import brotli
    compressions['BROTLI'] = brotli.compress
    decompressions['BROTLI'] = brotli.decompress
except ImportError:
    pass
try:
    import lz4.frame
    compressions['LZ4'] = lz4.frame.compress
    decompressions['LZ4'] = lz4.frame.decompress
except ImportError:
    pass

compressions = {k.upper(): v for k, v in compressions.items()}
decompressions = {k.upper(): v for k, v in decompressions.items()}

rev_map = {getattr(parquet_thrift.CompressionCodec, key): key for key in
           dir(parquet_thrift.CompressionCodec) if key in
           ['UNCOMPRESSED', 'SNAPPY', 'GZIP', 'LZO', 'BROTLI', 'LZ4']}


def compress_data(data, compression='gzip'):
    if isinstance(compression, dict):
        algorithm = compression.get('type', 'gzip')
        if isinstance(algorithm, int):
            algorithm = rev_map[compression]
        args = compression.get('args', None)
    else:
        algorithm = compression
        args = None

    if isinstance(algorithm, int):
        algorithm = rev_map[compression]

    if algorithm.upper() not in compressions:
        raise RuntimeError("Compression '%s' not available.  Options: %s" %
                (algorithm, sorted(compressions)))
    if args is None:
        return compressions[algorithm.upper()](data)
    else:
        if not isinstance(args, dict):
            raise ValueError("args dict entry is not a dict")
        return compressions[algorithm.upper()](data, **args)

def decompress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm.upper() not in decompressions:
        raise RuntimeError("Decompression '%s' not available.  Options: %s" %
                (algorithm.upper(), sorted(decompressions)))
    return decompressions[algorithm.upper()](data)
