import numpy as np
from lab_kernels.module2_memory_hierarchy import benchmark_strided_access, stencil_1d_shared


def test_benchmark_strided_access_copy_contract():
    x = np.arange(1024, dtype=np.float32)
    out = np.zeros(1024, dtype=np.float32)
    benchmark_strided_access[4, 256](x, out, 1)
    np.testing.assert_array_equal(out, x)


def test_benchmark_strided_access_stride():
    x = np.arange(1024, dtype=np.float32)
    out = np.zeros(128, dtype=np.float32)
    benchmark_strided_access[1, 256](x, out, 8)
    np.testing.assert_array_equal(out, x[::8][:128])


def test_stencil_1d_shared_nearest_edges():
    x = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    radius = 2
    out = np.zeros_like(x)
    block_size = 256
    blocks = (len(x) + block_size - 1) // block_size
    stencil_1d_shared[blocks, block_size, 0, (block_size + 2*radius)*4](x, out, radius)
    padded = np.pad(x, (radius, radius), mode="edge")
    expected = np.array([padded[i:i+2*radius+1].mean() for i in range(len(x))], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-4)
