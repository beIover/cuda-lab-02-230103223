
import numpy as np
from numba import cuda


@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:

        # Boundary clamping
        if idx == 0:
            left = d_in[idx]
            center = d_in[idx]
            right = d_in[idx + 1]

        elif idx == N - 1:
            left = d_in[idx - 1]
            center = d_in[idx]
            right = d_in[idx]

        else:
            left = d_in[idx - 1]
            center = d_in[idx]
            right = d_in[idx + 1]

        # 3-point smoothing
        d_out[idx] = (
            0.25 * left
            + 0.5 * center
            + 0.25 * right
        )


def run_stencil(h_in):

    N = h_in.size

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(h_in)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    stencil_1d[blocks_per_grid, threads_per_block](d_in, d_out, N)

    cuda.synchronize()

    return d_out.copy_to_host()


def cpu_stencil(arr):

    padded = np.pad(arr, (1, 1), mode='edge')

    return (
        0.25 * padded[:-2]
        + 0.5 * padded[1:-1]
        + 0.25 * padded[2:]
    )


if __name__ == "__main__":

    N = 10007

    h_in = np.sin(np.linspace(0, 10, N)).astype(np.float32)

    h_gpu = run_stencil(h_in)
    h_cpu = cpu_stencil(h_in)

    max_delta = np.max(np.abs(h_gpu - h_cpu))

    print("=" * 60)
    print("TASK 2: 1D STENCIL & CLAMPING")
    print("=" * 60)
    print(f"N = {N}")
    print(f"MAX DELTA = {max_delta}")

    assert np.allclose(h_gpu, h_cpu, atol=1e-4)

    print("TASK 2 PASSED")
    print("=" * 60)
