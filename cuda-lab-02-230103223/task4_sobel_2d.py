
import numpy as np
from numba import cuda


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):

    col, row = cuda.grid(2)

    # Boundary guard
    if row >= rows or col >= cols:
        return

    # Outer border must be zero
    if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
        d_out[row, col] = 0.0
        return

    # Sobel-X kernel:
    #
    # -1  0  1
    # -2  0  2
    # -1  0  1

    value = (
        -1.0 * d_in[row - 1, col - 1]
        + 1.0 * d_in[row - 1, col + 1]
        - 2.0 * d_in[row, col - 1]
        + 2.0 * d_in[row, col + 1]
        - 1.0 * d_in[row + 1, col - 1]
        + 1.0 * d_in[row + 1, col + 1]
    )

    d_out[row, col] = value


def run_sobel(h_img, threads_2d=(16, 16)):

    rows, cols = h_img.shape

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(h_img)

    blocks_x = (cols + threads_2d[0] - 1) // threads_2d[0]
    blocks_y = (rows + threads_2d[1] - 1) // threads_2d[1]

    blocks_2d = (blocks_x, blocks_y)

    sobel_x_kernel[
        blocks_2d,
        threads_2d
    ](
        d_in,
        d_out,
        rows,
        cols
    )

    cuda.synchronize()

    return d_out.copy_to_host()


if __name__ == "__main__":

    rows = 64
    cols = 64
    threads_2d = (16, 16)

    test_img = np.ones((rows, cols), dtype=np.float32)

    result = run_sobel(test_img, threads_2d)

    interior_max = np.max(
        np.abs(result[1:-1, 1:-1])
    )

    top_border_ok = np.all(result[0, :] == 0.0)
    bottom_border_ok = np.all(result[-1, :] == 0.0)
    left_border_ok = np.all(result[:, 0] == 0.0)
    right_border_ok = np.all(result[:, -1] == 0.0)

    print("=" * 60)
    print("TASK 4: 2D SOBEL-X")
    print("=" * 60)

    print(f"Image shape = {test_img.shape}")
    print(f"Threads per block = {threads_2d}")
    print(f"Interior maximum = {interior_max}")

    print(f"Top border zero = {top_border_ok}")
    print(f"Bottom border zero = {bottom_border_ok}")
    print(f"Left border zero = {left_border_ok}")
    print(f"Right border zero = {right_border_ok}")

    assert interior_max < 1e-5
    assert top_border_ok
    assert bottom_border_ok
    assert left_border_ok
    assert right_border_ok

    print("TASK 4 PASSED")
    print("=" * 60)
