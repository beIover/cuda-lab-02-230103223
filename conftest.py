import pytest
from numba import cuda


@pytest.fixture(scope="session", autouse=True)
def require_cuda():
    if not cuda.is_available():
        pytest.skip("CUDA GPU is not available; run these tests in a GPU runtime.")
