"""Small local pytest runner; detailed official grading harness was not supplied."""
import sys
import pytest


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "tests", *sys.argv[1:]]))
