"""Durable event-time join job.

Storage model
-------------
SQLite holds three kinds of data:

* join state: per-partition watermarks and the *active* left/right events,
* the committed batch log (one row per operation, including empty batches),
* the consumption position (byte offset into the input log).

One SQLite transaction per operation covers all three, so a crash can only
leave the job at an operation boundary.  The output JSONL file is a derived
export: it is rewritten from the committed batches whenever a run starts, so
an append that raced a crash can never duplicate a batch.

The active-event tables are bounded by the event-time windows, not by history:
a left event is removed once the right effective watermark reaches
``t + 5`` (after any required ``right=null`` row was emitted), and a right
event is removed once the left effective watermark reaches ``t + 5``.  Any
future opposite-side event must have ``t`` strictly greater than its own
partition watermark, hence strictly greater than the effective watermark, so
once the effective watermark passes ``t + 5`` no future pairing is possible.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .model import (
    MATCH_WINDOW,
    INITIAL_WATERMARK,
    INT64_MAX,
    INT64_MIN,
    InvalidOperation,
    Operation,
    decode_json_record,
    event_dict,
    is_token,
    parse_operation,
    serialize_batch,
)

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
CREATE TABLE partitions (
    side TEXT NOT NULL,
    partition TEXT NOT NULL,
    watermark INTEGER NOT NULL,
    PRIMARY KEY (side, partition)
);
CREATE TABLE left_events (
    id TEXT PRIMARY KEY,
    partition TEXT NOT NULL,
    key TEXT NOT NULL,
    t INTEGER NOT NULL,
    value_json TEXT NOT NULL,
    matched INTEGER NOT NULL
);
CREATE INDEX left_events_key_t ON left_events (key, t);
CREATE INDEX left_events_t ON left_events (t);
CREATE TABLE right_events (
    id TEXT PRIMARY KEY,
    partition TEXT NOT NULL,
    key TEXT NOT NULL,
    t INTEGER NOT NULL,
    value_json TEXT NOT NULL
);
CREATE INDEX right_events_key_t ON right_events (key, t);
CREATE INDEX right_events_t ON right_events (t);
CREATE TABLE seen_ids (
    id TEXT PRIMARY KEY
);
CREATE TABLE batches (
    seq INTEGER PRIMARY KEY,
    payload TEXT NOT NULL
);
"""


@dataclass
class RunResult:
    batches: int
    stopped_seq: int
    invalid: Optional[InvalidOperation]
    reached_eof: bool
    unterminated_tail: bool


def _window_bounds(t: int) -> tuple[int, int]:
    """Inclusive join window, clamped so SQLite binds stay inside int64."""
    lo = max(t - MATCH_WINDOW, INT64_MIN)
    hi = min(t + MATCH_WINDOW, INT64_MAX)
    return lo, hi


def normalize_partitions(values: Any, label: str) -> list[str]:
    """Validate a non-empty set of partition identifiers."""
    if isinstance(values, str):
        values = [v for v in values.split(",") if v != ""]
    parts = list(values)
    if not parts:
        raise ValueError(f"{label} partitions must be non-empty")
    for p in parts:
        if not is_token(p):
            raise ValueError(f"invalid {label} partition token {p!r}")
    return sorted(set(parts))


class JoinJob:
    """One durable join job over one SQLite database."""

    def __init__(self, db_path: os.PathLike[str] | str) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(
            str(self.db_path), isolation_level=None, timeout=30.0
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        try:
            version = self._meta("schema_version")
        except (sqlite3.Error, RuntimeError):
            self.conn.close()
            raise RuntimeError(
                f"{self.db_path} is not an initialized join database"
            ) from None
        if version != SCHEMA_VERSION:
            self.conn.close()
            raise RuntimeError(
                f"{self.db_path} has schema version {version!r}, "
                f"expected {SCHEMA_VERSION!r}"
            )
        self.left_partitions: list[str] = json.loads(self._meta("left_partitions"))
        self.right_partitions: list[str] = json.loads(self._meta("right_partitions"))
        self.next_seq = int(self._meta("next_seq"))
        self.consumed_offset = int(self._meta("consumed_offset"))
        self.fault_hook: Optional[Callable[[str, int], None]] = None
        self._output_fh = None

    # ------------------------------------------------------------------ init
    @classmethod
    def initialize(
        cls,
        db_path: os.PathLike[str] | str,
        left_partitions: Any,
        right_partitions: Any,
    ) -> "JoinJob":
        path = Path(db_path)
        if path.exists() and path.stat().st_size > 0:
            raise FileExistsError(f"{path} already exists; refusing to reinitialize")
        left = normalize_partitions(left_partitions, "left")
        right = normalize_partitions(right_partitions, "right")
        # Build the database at a sibling path and rename it into place, so a
        # crash during initialization never leaves a half-written job file.
        tmp = path.with_name(f"{path.name}.init-{os.getpid()}")
        for leftover in (
            tmp,
            Path(str(tmp) + "-wal"),
            Path(str(tmp) + "-shm"),
            Path(str(tmp) + "-journal"),
        ):
            if leftover.exists():
                leftover.unlink()
        try:
            conn = sqlite3.connect(str(tmp), isolation_level=None, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            # executescript commits implicitly; meta rows are written inside
            # one explicit transaction and mark the database complete.
            conn.executescript(_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            for side, parts in (("left", left), ("right", right)):
                for p in parts:
                    conn.execute(
                        "INSERT INTO partitions(side, partition, watermark) "
                        "VALUES (?,?,?)",
                        (side, p, INITIAL_WATERMARK),
                    )
            meta = {
                "schema_version": SCHEMA_VERSION,
                "left_partitions": json.dumps(left),
                "right_partitions": json.dumps(right),
                "next_seq": "1",
                "consumed_offset": "0",
            }
            for k, v in meta.items():
                conn.execute("INSERT INTO meta(k, v) VALUES (?,?)", (k, v))
            conn.execute("COMMIT")
            # Fold the WAL into the main file so the rename moves everything.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            os.replace(tmp, path)
        except BaseException:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            for leftover in (
                tmp,
                Path(str(tmp) + "-wal"),
                Path(str(tmp) + "-shm"),
                Path(str(tmp) + "-journal"),
            ):
                if leftover.exists():
                    leftover.unlink()
            raise
        return cls(path)

    # --------------------------------------------------------------- helpers
    def _meta(self, key: str) -> str:
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        if row is None:
            raise RuntimeError(f"missing meta key {key!r}")
        return row[0]

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("UPDATE meta SET v=? WHERE k=?", (value, key))

    def _partition_watermark(self, side: str, partition: str, seq: int) -> int:
        row = self.conn.execute(
            "SELECT watermark FROM partitions WHERE side=? AND partition=?",
            (side, partition),
        ).fetchone()
        if row is None:
            raise InvalidOperation(
                seq, f"unknown partition {partition!r} on side {side!r}"
            )
        return int(row[0])

    def _effective(self, side: str) -> int:
        # Minimum over *all* initialized partitions; idle partitions never
        # infer progress.
        row = self.conn.execute(
            "SELECT MIN(watermark) FROM partitions WHERE side=?", (side,)
        ).fetchone()
        assert row is not None and row[0] is not None
        return int(row[0])

    def _fault(self, point: str, seq: int) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point, seq)
        spec = os.environ.get("JOINJOB_FAULT")
        if not spec:
            return
        point_name, _, seq_text = spec.partition(":")
        if point_name == point and seq_text.isdigit() and int(seq_text) == seq:
            os._exit(137)  # hard process death, no cleanup

    # ------------------------------------------------------------ row mappers
    @staticmethod
    def _left_event(row: tuple) -> dict[str, Any]:
        (event_id, partition, key, t, value_json, _matched) = row
        return event_dict("left", partition, event_id, key, t, json.loads(value_json))

    @staticmethod
    def _right_event(row: tuple) -> dict[str, Any]:
        (event_id, partition, key, t, value_json) = row
        return event_dict(
            "right", partition, event_id, key, t, json.loads(value_json)
        )

    @staticmethod
    def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # ASCII order by left id, then right id; a null right sorts first.
        rows.sort(
            key=lambda r: (
                r["left"]["id"],
                "" if r["right"] is None else r["right"]["id"],
            )
        )
        return rows

    # ------------------------------------------------------------- apply ops
    def _apply(self, op: Operation) -> list[dict[str, Any]]:
        if op.kind == "watermark":
            return self._apply_watermark(op)
        if op.side == "left":
            return self._apply_left_event(op)
        return self._apply_right_event(op)

    def _claim_id(self, op: Operation) -> None:
        # Event ids are globally unique across both sides and all time.  The
        # ledger is a dedup index, deliberately kept separate from join
        # state: join state is reclaimed by event time, this is not.
        assert op.event_id is not None
        if self.conn.execute(
            "SELECT 1 FROM seen_ids WHERE id=?", (op.event_id,)
        ).fetchone() is not None:
            raise InvalidOperation(op.seq, f"duplicate event id {op.event_id!r}")
        self.conn.execute("INSERT INTO seen_ids(id) VALUES (?)", (op.event_id,))

    def _apply_left_event(self, op: Operation) -> list[dict[str, Any]]:
        assert op.t is not None and op.event_id is not None and op.key is not None
        wm = self._partition_watermark("left", op.partition, op.seq)
        if op.t <= wm:
            raise InvalidOperation(
                op.seq,
                f"event t={op.t} is not greater than left/{op.partition} "
                f"watermark {wm}",
            )
        self._claim_id(op)
        left = event_dict("left", op.partition, op.event_id, op.key, op.t, op.value)
        lo, hi = _window_bounds(op.t)
        rights = self.conn.execute(
            "SELECT id, partition, key, t, value_json FROM right_events "
            "WHERE key=? AND t>=? AND t<=? ORDER BY id",
            (op.key, lo, hi),
        ).fetchall()
        rows = [{"left": left, "right": self._right_event(r)} for r in rights]
        matched = bool(rights)
        right_eff = self._effective("right")
        if right_eff < op.t + MATCH_WINDOW:
            self.conn.execute(
                "INSERT INTO left_events(id, partition, key, t, value_json, matched) "
                "VALUES (?,?,?,?,?,?)",
                (
                    op.event_id,
                    op.partition,
                    op.key,
                    op.t,
                    json.dumps(op.value, ensure_ascii=False),
                    1 if matched else 0,
                ),
            )
        elif not matched:
            # The right side is already past this event's window and nothing
            # received paired with it: the unmatched verdict is final now.
            rows.append({"left": left, "right": None})
        # Retained only when it can still pair; a matched event whose window
        # has closed needs no further state.
        return self._sorted_rows(rows)

    def _apply_right_event(self, op: Operation) -> list[dict[str, Any]]:
        assert op.t is not None and op.event_id is not None and op.key is not None
        wm = self._partition_watermark("right", op.partition, op.seq)
        if op.t <= wm:
            raise InvalidOperation(
                op.seq,
                f"event t={op.t} is not greater than right/{op.partition} "
                f"watermark {wm}",
            )
        self._claim_id(op)
        right = event_dict("right", op.partition, op.event_id, op.key, op.t, op.value)
        lo, hi = _window_bounds(op.t)
        # Matched lefts are included: a newly received sample still pairs with
        # them, and every (left, right) pair is distinct.
        lefts = self.conn.execute(
            "SELECT id, partition, key, t, value_json, matched FROM left_events "
            "WHERE key=? AND t>=? AND t<=? ORDER BY id",
            (op.key, lo, hi),
        ).fetchall()
        rows = [{"left": self._left_event(l), "right": right} for l in lefts]
        if lefts:
            self.conn.execute(
                "UPDATE left_events SET matched=1 WHERE key=? AND t>=? AND t<=?",
                (op.key, lo, hi),
            )
        if self._effective("left") < op.t + MATCH_WINDOW:
            self.conn.execute(
                "INSERT INTO right_events(id, partition, key, t, value_json) "
                "VALUES (?,?,?,?,?)",
                (
                    op.event_id,
                    op.partition,
                    op.key,
                    op.t,
                    json.dumps(op.value, ensure_ascii=False),
                ),
            )
        # Otherwise no future left event can fall inside this window.
        return self._sorted_rows(rows)

    def _apply_watermark(self, op: Operation) -> list[dict[str, Any]]:
        assert op.watermark is not None
        current = self._partition_watermark(op.side, op.partition, op.seq)
        if op.watermark < current:
            raise InvalidOperation(
                op.seq,
                f"watermark {op.watermark} would decrease {op.side}/{op.partition} "
                f"from {current}",
            )
        self.conn.execute(
            "UPDATE partitions SET watermark=? WHERE side=? AND partition=?",
            (op.watermark, op.side, op.partition),
        )
        rows: list[dict[str, Any]] = []
        if op.side == "right":
            right_eff = self._effective("right")
            cutoff = right_eff - MATCH_WINDOW
            undecided = self.conn.execute(
                "SELECT id, partition, key, t, value_json, matched FROM left_events "
                "WHERE matched=0 AND t<=? ORDER BY id",
                (cutoff,),
            ).fetchall()
            rows = [{"left": self._left_event(r), "right": None} for r in undecided]
            # Every left event at or below the cutoff is finished: unmatched
            # ones got their verdict above, matched ones have no future pair.
            self.conn.execute("DELETE FROM left_events WHERE t<=?", (cutoff,))
        else:
            left_eff = self._effective("left")
            self.conn.execute(
                "DELETE FROM right_events WHERE t<=?", (left_eff - MATCH_WINDOW,)
            )
        return self._sorted_rows(rows)

    @staticmethod
    def _operation_echo(op: Operation) -> dict[str, Any]:
        if op.kind == "watermark":
            return {
                "side": op.side,
                "partition": op.partition,
                "watermark": op.watermark,
            }
        return event_dict(
            op.side, op.partition, op.event_id, op.key, op.t, op.value
        )

    # ------------------------------------------------------------- commit op
    def _apply_and_commit(self, op: Operation, end_offset: int) -> str:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self._apply(op)
            payload = serialize_batch(
                {
                    "seq": op.seq,
                    "op": self._operation_echo(op),
                    "rows": rows,
                }
            )
            self.conn.execute(
                "INSERT INTO batches(seq, payload) VALUES (?,?)", (op.seq, payload)
            )
            self._set_meta("next_seq", str(op.seq + 1))
            self._set_meta("consumed_offset", str(end_offset))
            self._fault("before_commit", op.seq)
            self.conn.execute("COMMIT")
        except BaseException:
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        self.next_seq = op.seq + 1
        self.consumed_offset = end_offset
        self._fault("after_commit", op.seq)
        return payload

    # ----------------------------------------------------------- output file
    def export(self, output_path: os.PathLike[str] | str) -> int:
        """Rewrite the output JSONL from committed batches atomically."""
        target = Path(output_path)
        tmp = target.with_name(target.name + ".tmp")
        rows = self.conn.execute(
            "SELECT payload FROM batches ORDER BY seq"
        ).fetchall()
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            for (payload,) in rows:
                fh.write(payload)
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return len(rows)

    def _open_output(self, output_path: os.PathLike[str] | str) -> None:
        self.export(output_path)
        self._output_fh = open(output_path, "a", encoding="utf-8", newline="\n")

    def _append_output(self, payload: str) -> None:
        if self._output_fh is not None:
            self._output_fh.write(payload)
            self._output_fh.write("\n")

    def process_log(
        self,
        input_path: os.PathLike[str] | str,
        output_path: Optional[os.PathLike[str] | str] = None,
        max_ops: Optional[int] = None,
    ) -> RunResult:
        """Consume newline-terminated records from the committed byte offset.

        An unterminated tail is not consumed: it may still be mid-write, and
        its byte offset stays uncommitted so a later run reads it whole.
        """
        if max_ops is not None and max_ops < 0:
            raise ValueError("max_ops must be non-negative")
        source = Path(input_path)
        if not source.exists():
            raise FileNotFoundError(input_path)
        size = source.stat().st_size
        if size < self.consumed_offset:
            raise RuntimeError(
                f"input is shorter ({size} bytes) than committed offset "
                f"{self.consumed_offset}; the consumed prefix was altered"
            )
        if output_path is not None:
            self._open_output(output_path)
        written = 0
        invalid: Optional[InvalidOperation] = None
        reached_eof = True
        unterminated = False
        exhausted = False
        try:
            with open(source, "rb") as handle:
                handle.seek(self.consumed_offset)
                while max_ops is None or written < max_ops:
                    raw = handle.readline()
                    if not raw:
                        exhausted = True
                        break
                    if not raw.endswith(b"\n"):
                        unterminated = True
                        reached_eof = False
                        break
                    end_offset = handle.tell()
                    seq = self.next_seq
                    try:
                        record = decode_json_record(raw[:-1], seq)
                        op = parse_operation(record, seq)
                        payload = self._apply_and_commit(op, end_offset)
                    except InvalidOperation as exc:
                        invalid = exc
                        reached_eof = False
                        break
                    written += 1
                    self._append_output(payload)
                if not exhausted and invalid is None and not unterminated:
                    reached_eof = False  # stopped early at max_ops
        finally:
            if self._output_fh is not None:
                self._output_fh.close()
                self._output_fh = None
        return RunResult(
            batches=written,
            stopped_seq=self.next_seq,
            invalid=invalid,
            reached_eof=reached_eof,
            unterminated_tail=unterminated,
        )

    # -------------------------------------------------------------- inspect
    def replay_batches(self) -> list[dict[str, Any]]:
        return [
            json.loads(payload)
            for (payload,) in self.conn.execute(
                "SELECT payload FROM batches ORDER BY seq"
            ).fetchall()
        ]

    def state_summary(self) -> dict[str, Any]:
        partitions: dict[str, dict[str, int]] = {"left": {}, "right": {}}
        for side, partition, watermark in self.conn.execute(
            "SELECT side, partition, watermark FROM partitions "
            "ORDER BY side, partition"
        ).fetchall():
            partitions[side][partition] = watermark
        return {
            "next_seq": self.next_seq,
            "consumed_offset": self.consumed_offset,
            "batches": self.conn.execute(
                "SELECT COUNT(*) FROM batches"
            ).fetchone()[0],
            "left_events_retained": self.conn.execute(
                "SELECT COUNT(*) FROM left_events"
            ).fetchone()[0],
            "right_events_retained": self.conn.execute(
                "SELECT COUNT(*) FROM right_events"
            ).fetchone()[0],
            "left_partitions": self.left_partitions,
            "right_partitions": self.right_partitions,
            "partition_watermarks": partitions,
            "effective_watermarks": {
                "left": self._effective("left"),
                "right": self._effective("right"),
            },
        }

    def close(self) -> None:
        if self._output_fh is not None:
            self._output_fh.close()
            self._output_fh = None
        self.conn.close()

    def __enter__(self) -> "JoinJob":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
