"""Pure validation and event-time join rules.

Storage code deliberately does not live here.  The durable job and the
independent full-history reference model used for differential testing both
import the *contract* from this module, while their bookkeeping is written
separately, so agreement is not an accident of shared implementation.

Integer checks use ``type(value) is int``: Python booleans are a subclass of
``int`` and the contract forbids ``true``/``false`` wherever an integer is
required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

MATCH_WINDOW = 5  # seconds, inclusive: |left.t - right.t| <= 5
INITIAL_WATERMARK = -1
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
SIDES = ("left", "right")

_ASCII_TOKEN = re.compile(r"\A[0-9A-Za-z_]+\Z", re.ASCII)
_EVENT_REQUIRED = frozenset({"side", "partition", "id", "key", "t", "value"})
_WATERMARK_REQUIRED = frozenset({"side", "partition", "watermark"})
_SEQ = "seq"


class InvalidOperation(Exception):
    """A log record that violates the operation contract."""

    def __init__(self, seq: int, reason: str) -> None:
        super().__init__(f"seq {seq}: {reason}")
        self.seq = seq
        self.reason = reason


def is_token(value: object) -> bool:
    """Non-empty ASCII alphanumeric/underscore string."""
    return isinstance(value, str) and _ASCII_TOKEN.fullmatch(value) is not None


def is_i64(value: object) -> bool:
    return type(value) is int and INT64_MIN <= value <= INT64_MAX


def _reject_constant(text: str) -> Any:
    raise ValueError(f"non-finite JSON number {text}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate object key {key!r}")
        out[key] = value
    return out


def decode_json_record(raw: bytes, seq: int) -> dict[str, Any]:
    """Decode one UTF-8 JSONL record, rejecting ambiguous JSON."""
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InvalidOperation(seq, f"line is not valid UTF-8: {exc}") from None
    try:
        record = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise InvalidOperation(seq, f"line is not valid JSON: {exc}") from None
    if not isinstance(record, dict):
        raise InvalidOperation(seq, "operation must be a JSON object")
    return record


@dataclass(frozen=True)
class Operation:
    seq: int
    kind: str  # "event" | "watermark"
    side: str
    partition: str
    event_id: Optional[str] = None
    key: Optional[str] = None
    t: Optional[int] = None
    value: Any = None
    watermark: Optional[int] = None


def _check_fields(
    record: dict[str, Any], required: frozenset[str], seq: int
) -> None:
    allowed = required | {_SEQ}
    keys = frozenset(record)
    if not required <= keys or not keys <= allowed:
        missing = sorted(required - keys)
        extra = sorted(keys - allowed)
        raise InvalidOperation(
            seq, f"fields mismatch (missing={missing}, extra={extra})"
        )


def parse_operation(record: dict[str, Any], expected_seq: int) -> Operation:
    """Validate one operation and normalize it.

    ``seq`` is optional in the input because the contiguous log position is
    authoritative; when present it must equal that position.  This accepts
    both bare operations and operations that carry their sequence number.
    """
    if _SEQ in record:
        value = record[_SEQ]
        if isinstance(value, bool) or not is_i64(value):
            raise InvalidOperation(expected_seq, "seq must be a non-boolean integer")
        if value != expected_seq:
            raise InvalidOperation(
                expected_seq, f"seq is {value}, expected {expected_seq}"
            )

    if "watermark" in record:
        _check_fields(record, _WATERMARK_REQUIRED, expected_seq)
        side = record["side"]
        partition = record["partition"]
        watermark = record["watermark"]
        if side not in SIDES:
            raise InvalidOperation(expected_seq, f"invalid side {side!r}")
        if not is_token(partition):
            raise InvalidOperation(expected_seq, "partition must be an ASCII token")
        if isinstance(watermark, bool) or not is_i64(watermark):
            raise InvalidOperation(
                expected_seq,
                "watermark must be a non-boolean signed 64-bit integer",
            )
        if watermark < INITIAL_WATERMARK:
            raise InvalidOperation(
                expected_seq, "watermark must be greater than or equal to -1"
            )
        return Operation(
            seq=expected_seq,
            kind="watermark",
            side=side,
            partition=partition,
            watermark=watermark,
        )

    _check_fields(record, _EVENT_REQUIRED, expected_seq)
    side = record["side"]
    partition = record["partition"]
    event_id = record["id"]
    key = record["key"]
    t = record["t"]
    value = record["value"]

    if side not in SIDES:
        raise InvalidOperation(expected_seq, f"invalid side {side!r}")
    if not is_token(partition):
        raise InvalidOperation(expected_seq, "partition must be an ASCII token")
    if not is_token(event_id):
        raise InvalidOperation(expected_seq, "id must be an ASCII token")
    if not is_token(key):
        raise InvalidOperation(expected_seq, "key must be an ASCII token")
    if isinstance(t, bool) or not is_i64(t):
        raise InvalidOperation(
            expected_seq, "t must be a non-boolean signed 64-bit integer"
        )
    if t < 0:
        raise InvalidOperation(expected_seq, "t must be non-negative")
    if not (value is None or isinstance(value, str)):
        raise InvalidOperation(expected_seq, "value must be a string or null")

    return Operation(
        seq=expected_seq,
        kind="event",
        side=side,
        partition=partition,
        event_id=event_id,
        key=key,
        t=t,
        value=value,
    )


def event_dict(
    side: str,
    partition: str,
    event_id: str,
    key: str,
    t: int,
    value: Optional[str],
) -> dict[str, Any]:
    """A complete event object as it appears in result rows."""
    return {
        "side": side,
        "partition": partition,
        "id": event_id,
        "key": key,
        "t": t,
        "value": value,
    }


def result_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    """ASCII order by left id, then right id; null right sorts first."""
    right = row["right"]
    return (row["left"]["id"], "" if right is None else right["id"])


def serialize_batch(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )
