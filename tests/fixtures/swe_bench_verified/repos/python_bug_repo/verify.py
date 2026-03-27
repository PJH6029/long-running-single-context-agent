from __future__ import annotations

import sys

import bugs


EXPECTED = {
    "sbv-001": ("sbv_001", 1),
    "sbv-002": ("sbv_002", 2),
    "sbv-003": ("sbv_003", 3),
    "sbv-004": ("sbv_004", 4),
    "sbv-005": ("sbv_005", 5),
    "sbv-006": ("sbv_006", 6),
    "sbv-007": ("sbv_007", 7),
    "sbv-008": ("sbv_008", 8),
}


def main(task_id: str) -> int:
    symbol, expected = EXPECTED[task_id]
    value = getattr(bugs, symbol)()
    if value == expected:
        print(f"{task_id} passed")
        return 0
    print(f"{task_id} failed: expected {expected}, got {value}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

