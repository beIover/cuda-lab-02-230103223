
import numpy as np
from numba import cuda


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):

    N = h_arr.size

    d_arr = cuda.to_device(h_arr)

    threads_per_block = 256
    blocks_per_grid = 64

    grid_stride_scale_kernel[
        blocks_per_grid,
        threads_per_block
    ](d_arr, factor, N)

    cuda.synchronize()

    return d_arr.copy_to_host()


if __name__ == "__main__":

    N = 100000
    factor = 4.25

    h_arr = np.ones(N, dtype=np.float32)

    result = run_grid_stride(h_arr, factor)

    expected = np.full(N, factor, dtype=np.float32)

    max_delta = np.max(np.abs(result - expected))

    print("=" * 60)
    print("TASK 3: GRID-STRIDE VECTOR SCALING")
    print("=" * 60)

    print(f"N = {N}")
    print(f"Factor = {factor}")
    print(f"Threads per block = 256")
    print(f"Blocks per grid = 64")
    print(f"Total threads = {256 * 64}")
    print(f"MAX DELTA = {max_delta}")

    assert np.allclose(result, expected)

    print("TASK 3 PASSED")
    print("=" * 60)
