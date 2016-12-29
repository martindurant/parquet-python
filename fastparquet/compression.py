
import gzip
from .thrift_structures import parquet_thrift
# TODO: use stream/direct-to-buffer conversions instead of memcopy

# TODO: enable ability to pass kwargs to compressor

try:
    gzip_compress = gzip.compress
    gzip_decompress = gzip.decompress

except AttributeError:
    # Python < 3.2 doesn't have these functions; copy them here.
    import io

    def gzip_compress(data, compresslevel=9):
        buf = io.BytesIO()
        with gzip.GzipFile(
                fileobj=buf, mode='wb', compresslevel=compresslevel) as f:
            f.write(data)
        return buf.getvalue()

    def gzip_decompress(data):
        with GzipFile(fileobj=io.BytesIO(data)) as f:
            return f.read()


compressions = {'GZIP': gzip_compress,
                'UNCOMPRESSED': lambda x: x}

decompressions = {'GZIP': gzip_decompress,
                  'UNCOMPRESSED': lambda x: x}
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

compressions = {k.upper(): v for k, v in compressions.items()}
decompressions = {k.upper(): v for k, v in decompressions.items()}

rev_map = {getattr(parquet_thrift.CompressionCodec, key): key for key in
           dir(parquet_thrift.CompressionCodec) if key in compressions}


def compress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm.upper() not in compressions:
        raise RuntimeError("Compression '%s' not available.  Options: %s" %
                (algorithm, sorted(compressions)))
    return compressions[algorithm.upper()](data)


def decompress_data(data, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm.upper() not in decompressions:
        raise RuntimeError("Decompression '%s' not available.  Options: %s" %
                (algorithm.upper(), sorted(decompressions)))
    return decompressions[algorithm.upper()](data)
