"""Shared flight number parsing utilities."""
import re
from typing import Tuple

FLIGHT_NUMBER_RE = re.compile(r"^([A-Za-z]{2})\s*(\d+)$")


def parse_flight_number(query: str) -> Tuple[str, int]:
    """
    Parse inputs like 'AA1002', 'AA 1002', or 'aa1002'.
    Returns (carrier_code, flight_num).
    """
    cleaned = query.strip().upper().replace(" ", "")
    match = FLIGHT_NUMBER_RE.match(cleaned)
    if not match:
        raise ValueError(
            "Invalid flight number. Use carrier code + number, e.g. AA1002 or AA 1002."
        )
    return match.group(1), int(match.group(2))
