# tests/test_core.py

import pytest

from GKiPO.core import add_numbers


def test_add_numbers_success():
    """Test that add_numbers correctly adds positive integers."""
    result = add_numbers(2, 3)
    assert result == 5


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (0, 0, 0),
        (-1, 1, 0),
        (-5, -5, -10),
        (100, 200, 300),
    ],
)
def test_add_numbers_parametrized(a, b, expected):
    """Test add_numbers across various scenarios using parametrization."""
    assert add_numbers(a, b) == expected
