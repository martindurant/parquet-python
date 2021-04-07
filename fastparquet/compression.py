
import cramjam
from .thrift_structures import parquet_thrift

# TODO: use stream/direct-to-buffer conversions instead of memcopy

compressions = {
    'UNCOMPRESSED': lambda x: x
}
decompressions = {
    'UNCOMPRESSED': lambda x, y: x
}

# Gzip is present regardless
COMPRESSION_LEVEL = 6


def gzip_compress_v3(data, compresslevel=COMPRESSION_LEVEL):
    return cramjam.gzip.compress(data, level=compresslevel)


def gzip_decompress(data, uncompressed_size):
    return cramjam.gzip.decompress(data, output_len=uncompressed_size)


compressions['GZIP'] = gzip_compress_v3
decompressions['GZIP'] = gzip_decompress

def snappy_decompress(data, uncompressed_size):
    return cramjam.snappy.decompress_raw(data)
compressions['SNAPPY'] = cramjam.snappy.compress_raw
decompressions['SNAPPY'] = snappy_decompress
try:
    import lzo
    def lzo_decompress(data, uncompressed_size):
        return lzo.decompress(data)
    compressions['LZO'] = lzo.compress
    decompressions['LZO'] = lzo_decompress
except ImportError:
    pass
compressions['BROTLI'] = cramjam.brotli.compress
decompressions['BROTLI'] = cramjam.brotli.decompress

try:
    import lz4.block
    def lz4_compress(data, **kwargs):
        kwargs['store_size'] = False
        return lz4.block.compress(data, **kwargs)
    def lz4_decompress(data, uncompressed_size):
        return lz4.block.decompress(data, uncompressed_size=uncompressed_size)
    compressions['LZ4'] = lz4_compress
    decompressions['LZ4'] = lz4_decompress
except ImportError:
    pass
compressions['ZSTD'] = cramjam.zstd.compress
decompressions['ZSTD'] = cramjam.zstd.decompress

compressions = {k.upper(): v for k, v in compressions.items()}
decompressions = {k.upper(): v for k, v in decompressions.items()}

rev_map = {getattr(parquet_thrift.CompressionCodec, key): key for key in
           dir(parquet_thrift.CompressionCodec) if key in
           ['UNCOMPRESSED', 'SNAPPY', 'GZIP', 'LZO', 'BROTLI', 'LZ4', 'ZSTD']}


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

def decompress_data(data, uncompressed_size, algorithm='gzip'):
    if isinstance(algorithm, int):
        algorithm = rev_map[algorithm]
    if algorithm.upper() not in decompressions:
        raise RuntimeError("Decompression '%s' not available.  Options: %s" %
                (algorithm.upper(), sorted(decompressions)))
    return decompressions[algorithm.upper()](data, uncompressed_size)
