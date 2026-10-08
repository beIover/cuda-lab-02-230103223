
# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103223  
**Allocated GPU Node:** Tesla T4  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** 729E3023E018C902EE89  

---

## Task 1 — Warp Divergence Microbenchmark

Three CUDA kernels were benchmarked using 10 consecutive trials after a warm-up.

| Kernel | Description | Average Time |
|---|---|---:|
| Kernel A | Uniform Path | 0.002669 s |
| Kernel B | Full Divergence | 0.019509 s |
| Kernel C | Warp-Aligned Branching | 0.007533 s |

### Observation

Kernel A was the fastest because all threads followed the same execution path.

Kernel B was the slowest because adjacent threads in the same warp followed different branches, causing warp divergence.

Kernel C was faster than Kernel B because entire warps followed the same execution path.

---

## Task 2 — 1D Boundary Stencil & Halo Protection

The CUDA kernel implements a 3-point smoothing filter.

The boundary elements use clamping so that the first element uses itself as the left neighbor and the last element uses itself as the right neighbor.

Verification:

- N = 10007
- MAX DELTA = 5.960464477539063e-08
- TASK 2 PASSED

The GPU result matched the CPU reference within the required tolerance.

---

## Task 3 — Grid-Stride Vector Scaling

The vector was scaled using a CUDA grid-stride loop.

Configuration:

- N = 100000
- Factor = 4.25
- Threads per block = 256
- Blocks per grid = 64
- Total threads = 16384
- MAX DELTA = 0.0

The grid-stride loop allows the 16384 CUDA threads to process all 100000 elements.

**TASK 3 PASSED**

---

## Task 4 — 2D Sobel-X

A 2D Sobel-X CUDA kernel was tested using a 64 × 64 flat image.

Configuration:

- Image shape = (64, 64)
- Threads per block = (16, 16)
- Interior maximum = 0.0
- Top border zero = True
- Bottom border zero = True
- Left border zero = True
- Right border zero = True

Because the input image is flat, the interior Sobel-X gradient is zero.

**TASK 4 PASSED**

---

## Automated Verification

The official verification script completed successfully.

```text
[PASS] Task 2 (1D Stencil & Clamping)
[PASS] Task 3 (Grid-Stride Scaling)
[PASS] Task 4 (2D Sobel Horizontal)

VERIFICATION SUCCESSFUL

OFFICIAL SUBMISSION TOKEN:
729E3023E018C902EE89
