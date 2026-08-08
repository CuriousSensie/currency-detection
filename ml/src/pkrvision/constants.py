"""Shared domain constants."""

DENOMINATIONS: tuple[int, ...] = (10, 20, 50, 100, 500, 1000, 5000)
CLASS_TO_DENOMINATION: dict[int, int] = dict(enumerate(DENOMINATIONS))
DENOMINATION_TO_CLASS: dict[int, int] = {value: key for key, value in CLASS_TO_DENOMINATION.items()}
PROJECT_SEED = 20260908
