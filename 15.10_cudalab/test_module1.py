import numpy as np
import pytest
from numba import cuda
from lab_kernels.module1_execution_model import (
    vector_add_1d, vector_scale_grid_stride, matrix_add_2d,
    matrix_transpose_naive, volume_init_3d, rgb_to_grayscale,
    vector_lerp, vector_threshold,
)


@pytest.mark.parametrize("n", [1, 255, 256, 1024, 1000003])
def test_vector_add_1d(n):
    rng = np.random.default_rng(42)
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)
    out = np.zeros_like(a)
    vector_add_1d[(n + 255)//256, 256](a, b, out)
    np.testing.assert_allclose(out, a + b, atol=1e-5)


@pytest.mark.parametrize("n", [1, 257, 10003])
def test_vector_scale_grid_stride(n):
    x = np.linspace(-2, 2, n, dtype=np.float32)
    out = np.zeros_like(x)
    vector_scale_grid_stride[64, 256](x, np.float32(2.5), out)
    np.testing.assert_allclose(out, x * 2.5, atol=1e-5)


def test_matrix_add_2d_non_square():
    rng = np.random.default_rng(7)
    A = rng.random((513, 1021), dtype=np.float32)
    B = rng.random((513, 1021), dtype=np.float32)
    C = np.zeros_like(A)
    matrix_add_2d[((1021+15)//16, (513+15)//16), (16,16)](A, B, C)
    np.testing.assert_allclose(C, A+B, atol=1e-5)


@pytest.mark.parametrize("shape", [(3, 5), (17, 31), (256, 129)])
def test_matrix_transpose(shape):
    m, n = shape
    A = np.arange(m*n, dtype=np.float32).reshape(m, n)
    B = np.zeros((n, m), dtype=np.float32)
    matrix_transpose_naive[((n+15)//16, (m+15)//16), (16,16)](A, B)
    np.testing.assert_allclose(B, A.T, atol=1e-5)


def test_volume_init_3d():
    D, H, W = 33, 47, 65
    vol = np.zeros((D,H,W), dtype=np.int64)
    volume_init_3d[((W+7)//8, (H+7)//8, (D+7)//8), (8,8,8)](vol, D, H, W)
    z, y, x = np.indices((D,H,W))
    expected = z*10000 + y*100 + x
    np.testing.assert_array_equal(vol, expected)


def test_rgb_to_grayscale():
    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, size=(37, 51, 3), dtype=np.uint8)
    gray = np.zeros((37,51), dtype=np.float32)
    rgb_to_grayscale[((51+15)//16, (37+15)//16), (16,16)](img, gray)
    expected = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    np.testing.assert_allclose(gray, expected, atol=1e-4)


def test_vector_lerp():
    rng = np.random.default_rng(10)
    n = 10007
    a = rng.random(n, dtype=np.float32)
    b = rng.random(n, dtype=np.float32)
    t = rng.random(n, dtype=np.float32)
    out = np.zeros_like(a)
    vector_lerp[(n+255)//256, 256](a, b, t, out)
    np.testing.assert_allclose(out, (1-t)*a+t*b, atol=1e-5)


def test_vector_threshold():
    x = np.array([-1, 0, 0.5, 1, 2], dtype=np.float32)
    mask = np.zeros(x.shape, dtype=np.uint8)
    vector_threshold[1, 256](x, np.float32(0.5), mask)
    np.testing.assert_array_equal(mask, (x >= 0.5).astype(np.uint8))
