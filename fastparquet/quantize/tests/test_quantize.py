import numpy as np
from fastparquet import quantize

def test_quantize_float32():
    seed = 2454389
    rng = np.random.RandomState(seed=seed)
    data = rng.normal(size=100000).astype(np.float32)
    buff = quantize.quantize_and_compress_float32(data, 0.1, 0.0, seed=1234)
    check_data = quantize.dequantize_and_decompress(buff)
    std = np.std(data - check_data)
    assert std < 0.05, "RICE float quantization and compression for float 32 is broken!"

def test_quantize_float32():
    seed = 2454389
    rng = np.random.RandomState(seed=seed)
    data = rng.normal(size=100000).astype(np.float64)
    buff = quantize.quantize_and_compress_float64(data, 0.1, 0.0, seed=1234)
    check_data = quantize.dequantize_and_decompress(buff)
    std = np.std(data - check_data)
    assert std < 0.05, "RICE float quantization and compression for float 64 is broken!"
