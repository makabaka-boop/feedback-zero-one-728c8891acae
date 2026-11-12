"""Independent full-history reference model for differential testing.

This model keeps *every* event it has ever seen, so it shares no incremental
state or eviction logic with the durable job.  Given the same operation log
it must produce equivalent batches.  Only the contract constants and the
validation helpers are imported from ``joinjob.model``; matching, watermark
arithmetic and ordering are re-derived here deliberately.

Each operation's batch is computed by rescanning the whole history:

* a left event pairs immediately with every received right event in the same
  key within the inclusive five-second window;
* a right event pairs immediately with every received, not yet decided left
  event in window (matched lefts included, so repeat samples keep pairing);
* ``right: null`` is emitted for a left event exactly when it has never
  matched and the effective right watermark is at least ``left.t + 5``.

A pair is keyed by (left id, right id), so no pair is emitted twice, and a
matched left is never given a null verdict.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from joinjob.model import (
    MATCH_WINDOW,
    Operation,
    InvalidOperation,
    decode_json_record,
    parse_operation,
)


class FullHistoryModel:
    def __init__(
        self, left_partitions: Iterable[str], right_partitions: Iterable[str]
    ) -> None:
        self.watermarks: dict[tuple[str, str], int] = {
            ("left", p): -1 for p in left_partitions
        }
        self.watermarks.update({("right", p): -1 for p in right_partitions})
        self.lefts: list[Operation] = []
        self.rights: list[Operation] = []
        self.ids_seen: set[str] = set()
        self.emitted_pairs: set[tuple[str, str]] = set()
        self.matched_lefts: set[str] = set()
        self.decided_lefts: set[str] = set()

    # ---------------------------------------------------------------- apply
    def apply(self, op: Operation) -> dict[str, Any]:
        if op.kind == "watermark":
            key = (op.side, op.partition)
            if key not in self.watermarks:
                raise InvalidOperation(op.seq, f"unknown partition {op.partition!r}")
            if op.watermark < self.watermarks[key]:
                raise InvalidOperation(
                    op.seq,
                    f"partition watermark {op.side}/{op.partition} would decrease "
                    f"from {self.watermarks[key]} to {op.watermark}",
                )
            self.watermarks[key] = op.watermark
        else:
            assert op.event_id is not None
            key = (op.side, op.partition)
            if key not in self.watermarks:
                raise InvalidOperation(op.seq, f"unknown partition {op.partition!r}")
            if op.event_id in self.ids_seen:
                raise InvalidOperation(op.seq, f"duplicate event id {op.event_id!r}")
            if op.t <= self.watermarks[key]:
                raise InvalidOperation(
                    op.seq,
                    f"event t={op.t} is not greater than {op.side}/{op.partition} "
                    f"watermark {self.watermarks[key]}",
                )
            self.ids_seen.add(op.event_id)
            (self.lefts if op.side == "left" else self.rights).append(op)

        return self._batch(op)

    # -------------------------------------------------------------- matching
    @staticmethod
    def _within(a: Operation, b: Operation) -> bool:
        return a.key == b.key and abs(a.t - b.t) <= MATCH_WINDOW

    @staticmethod
    def _event(op: Operation) -> dict[str, Any]:
        return {
            "side": op.side,
            "partition": op.partition,
            "id": op.event_id,
            "key": op.key,
            "t": op.t,
            "value": op.value,
        }

    def _effective(self, side: str) -> int:
        return min(w for (s, _p), w in self.watermarks.items() if s == side)

    def _pair_rows(self, left: Operation) -> list[dict[str, Any]]:
        rows = []
        for right in self.rights:
            pair = (left.event_id, right.event_id)
            if pair in self.emitted_pairs or not self._within(left, right):
                continue
            self.emitted_pairs.add(pair)
            self.matched_lefts.add(left.event_id)
            rows.append({"left": self._event(left), "right": self._event(right)})
        return rows

    def _null_row(self, left: Operation) -> Optional[dict[str, Any]]:
        if left.event_id in self.matched_lefts:
            return None
        if left.event_id in self.decided_lefts:
            return None
        if self._effective("right") < left.t + MATCH_WINDOW:
            return None
        self.decided_lefts.add(left.event_id)
        return {"left": self._event(left), "right": None}

    def _candidate_lefts(self, op: Operation) -> list[Operation]:
        if op.kind == "event" and op.side == "left":
            return [e for e in self.lefts if e.event_id == op.event_id]
        # Right events and watermarks can only act on already received lefts;
        # decided ones can no longer match.
        return [e for e in self.lefts if e.event_id not in self.decided_lefts]

    def _batch(self, op: Operation) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for left in self._candidate_lefts(op):
            pairs = self._pair_rows(left)
            rows.extend(pairs)
            if not pairs:
                null = self._null_row(left)
                if null is not None:
                    rows.append(null)
        rows.sort(
            key=lambda r: (
                r["left"]["id"],
                "" if r["right"] is None else r["right"]["id"],
            )
        )
        return {
            "seq": op.seq,
            "op": self._echo(op),
            "rows": rows,
        }

    def expected_retention(self) -> tuple[int, int]:
        """How many events the engine may still hold, derived from history.

        A left event stays live while ``t > right_eff - 5`` and a right event
        while ``t > left_eff - 5``; anything at or below that bound can never
        pair with a future opposite-side event, because future events are
        strictly above their own partition watermark, hence above the
        effective minimum.
        """
        live_left = sum(
            1 for e in self.lefts
            if e.t > self._effective("right") - MATCH_WINDOW
        )
        live_right = sum(
            1 for e in self.rights
            if e.t > self._effective("left") - MATCH_WINDOW
        )
        return live_left, live_right

    @staticmethod
    def _echo(op: Operation) -> dict[str, Any]:
        if op.kind == "watermark":
            return {
                "side": op.side,
                "partition": op.partition,
                "watermark": op.watermark,
            }
        return {
            "side": op.side,
            "partition": op.partition,
            "id": op.event_id,
            "key": op.key,
            "t": op.t,
            "value": op.value,
        }


def run_model_bytes(
    log_bytes: bytes, left_partitions: Iterable[str], right_partitions: Iterable[str]
) -> tuple[list[dict[str, Any]], Optional[InvalidOperation]]:
    """Replay a whole log; returns (batches, first invalid operation or None)."""
    model = FullHistoryModel(left_partitions, right_partitions)
    batches: list[dict[str, Any]] = []
    for index, raw in enumerate(log_bytes.splitlines(keepends=True), start=1):
        if not raw.endswith(b"\n"):
            break  # an unterminated tail is not part of the consumed prefix
        line = raw[:-1]
        try:
            record = decode_json_record(line, index)
            op = parse_operation(record, index)
            batch = model.apply(op)
        except InvalidOperation as exc:
            return batches, exc
        batches.append(batch)
    return batches, None
