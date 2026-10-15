"""Tasks 1–8 from the supplied CUDA Python lab specification."""
import numba
from numba import cuda, float32, uint8


@cuda.jit
def vector_add_1d(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] + b[i]


@cuda.jit
def vector_scale_grid_stride(x, alpha, out):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.size:
        out[i] = x[i] * alpha
        i += stride


@cuda.jit
def matrix_add_2d(A, B, C):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


@cuda.jit
def matrix_transpose_naive(A, B):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row < A.shape[0] and col < A.shape[1]:
        B[col, row] = A[row, col]


@cuda.jit
def volume_init_3d(vol, D, H, W):
    x, y, z = cuda.grid(3)
    if z < D and y < H and x < W:
        vol[z, y, x] = z * 10000 + y * 100 + x


@cuda.jit
def rgb_to_grayscale(img_rgb, img_gray):
    y, x = cuda.grid(2)
    if y < img_gray.shape[0] and x < img_gray.shape[1]:
        r = float32(img_rgb[y, x, 0])
        g = float32(img_rgb[y, x, 1])
        b = float32(img_rgb[y, x, 2])
        img_gray[y, x] = 0.299 * r + 0.587 * g + 0.114 * b


@cuda.jit
def vector_lerp(a, b, t, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (1.0 - t[i]) * a[i] + t[i] * b[i]


@cuda.jit
def vector_threshold(x, thresh, mask):
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < x.size:
        mask[i] = 1 if x[i] >= thresh else 0
        i += stride
