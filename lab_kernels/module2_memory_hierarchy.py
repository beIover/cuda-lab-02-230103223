"""Tasks 9–10. Task 9's source specification leaves the stride output contract ambiguous."""
from numba import cuda, float32


@cuda.jit
def benchmark_strided_access(in_arr, out_arr, stride):
    """Copy in_arr[i * stride] to out_arr[i] where the source index is in bounds.

    For stride=1 this is a contiguous copy. For stride>1, out_arr is expected
    to have ceil(len(in_arr)/stride) elements (or fewer, if explicitly desired).
    """
    i = cuda.grid(1)
    src = i * stride
    if i < out_arr.size and src < in_arr.size:
        out_arr[i] = in_arr[src]


@cuda.jit
def stencil_1d_shared(in_arr, out_arr, radius):
    """Nearest-edge moving average using dynamic shared memory.

    Launch with sharedmem=(block_size + 2*radius)*4 bytes.
    """
    shared = cuda.shared.array(0, dtype=float32)
    tid = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    base = cuda.blockIdx.x * block_size
    n = in_arr.size
    tile_len = block_size + 2 * radius

    # Each thread loads one or more positions, including halo.
    j = tid
    while j < tile_len:
        source_idx = base + j - radius
        if source_idx < 0:
            source_idx = 0
        elif source_idx >= n:
            source_idx = n - 1
        shared[j] = in_arr[source_idx]
        j += block_size

    cuda.syncthreads()

    i = base + tid
    if i < n:
        total = 0.0
        for k in range(2 * radius + 1):
            total += shared[tid + k]
        out_arr[i] = total / (2 * radius + 1)
