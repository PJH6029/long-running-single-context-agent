from __future__ import annotations

import sys

import pipeline


EXPECTED = {
    "sci-001": ("ci_001", 101),
    "sci-002": ("ci_002", 102),
    "sci-003": ("ci_003", 103),
    "sci-004": ("ci_004", 104),
    "sci-005": ("ci_005", 105),
    "sci-006": ("ci_006", 106),
    "sci-007": ("ci_007", 107),
    "sci-008": ("ci_008", 108),
}


def main(task_id: str) -> int:
    symbol, expected = EXPECTED[task_id]
    value = getattr(pipeline, symbol)()
    if value == expected:
        print(f"{task_id} passed")
        return 0
    print(f"{task_id} failed: expected {expected}, got {value}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

