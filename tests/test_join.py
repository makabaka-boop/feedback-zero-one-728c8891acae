"""Acceptance tests for the durable event-time join job.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joinjob.engine import JoinJob
from joinjob.model import InvalidOperation, parse_operation, serialize_batch
from tests.fuzz import (
    LEFT_PARTS,
    RIGHT_PARTS,
    encode_log,
    generate_edge_log,
    generate_log,
    run_engine,
    write_bytes,
    write_log,
)
from tests.reference_model import FullHistoryModel, run_model_bytes


def predicted_retention(
    records: list[dict[str, Any]]
) -> tuple[int, int]:
    """Live-event counts predicted by the full-history model."""
    model = FullHistoryModel(LEFT_PARTS, RIGHT_PARTS)
    for index, record in enumerate(records, start=1):
        op = parse_operation(record, index)
        model.apply(op)
    return model.expected_retention()

LEFT = ["lp1", "lp2", "lp3"]
RIGHT = ["rp1", "rp2", "rp3"]


def _ev(
    side: str,
    part: str,
    event_id: str,
    key: str,
    t: int,
    value: Optional[str] = None,
    seq: Optional[int] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "side": side,
        "partition": part,
        "id": event_id,
        "key": key,
        "t": t,
        "value": value,
    }
    if seq is not None:
        record["seq"] = seq
    return record


def _wm(
    side: str, part: str, watermark: int, seq: Optional[int] = None
) -> dict[str, Any]:
    record: dict[str, Any] = {"side": side, "partition": part, "watermark": watermark}
    if seq is not None:
        record["seq"] = seq
    return record


def _read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().splitlines()


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _read_batches(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _read_lines(path) if line.strip()]


def _rows_of(batch: dict[str, Any]) -> list[tuple[str, Optional[str]]]:
    return [
        (row["left"]["id"], None if row["right"] is None else row["right"]["id"])
        for row in batch["rows"]
    ]


class JobCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ws = tempfile.mkdtemp(prefix="joinjob-test-")
        self.db = os.path.join(self.ws, "job.db")
        self.log = os.path.join(self.ws, "ops.jsonl")
        self.out = os.path.join(self.ws, "batches.jsonl")

    def start(
        self,
        records: list[dict[str, Any]],
        left: list[str] = LEFT,
        right: list[str] = RIGHT,
    ) -> JoinJob:
        write_log(self.log, records)
        JoinJob.initialize(self.db, left, right)
        job = JoinJob(self.db)
        job.process_log(self.log, self.out)
        return job

    def run_job(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        job = self.start(records)
        try:
            return job.replay_batches()
        finally:
            job.close()

    def assert_matches_model(self, records: list[dict[str, Any]]) -> None:
        batches = self.run_job(records)
        model, error = run_model_bytes(encode_log(records), LEFT, RIGHT)
        self.assertIsNone(error)
        self.assertEqual(batches, model)


class TestMatchSemantics(JobCase):
    def test_five_second_boundary_inclusive(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 100),
            _ev("left", "lp1", "l2", "k", 101),
            _ev("right", "rp1", "r1", "k", 105),  # exactly +5 from l1, +4 from l2
            _ev("right", "rp1", "r2", "k", 106),  # +6 from l1 -> no pair
            _ev("right", "rp1", "r3", "k", 94),   # -6 from l1 -> no pair
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[2]), [("l1", "r1"), ("l2", "r1")])
        # 106 is +6 from l1 (outside) but exactly +5 from l2 (inside).
        self.assertEqual(_rows_of(batches[3]), [("l2", "r2")])
        self.assertEqual(_rows_of(batches[4]), [])

    def test_exactly_five_below_and_above(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("right", "rp1", "r1", "k", 5),   # |10-5| = 5 -> pair
            _ev("right", "rp1", "r2", "k", 4),   # |10-4| = 6 -> no pair
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[1]), [("l1", "r1")])
        self.assertEqual(_rows_of(batches[2]), [])

    def test_key_must_match(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "a", 10),
            _ev("right", "rp1", "r1", "b", 11),
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[1]), [])

    def test_duplicate_samples_are_kept_and_pairs_unique(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("left", "lp1", "l1b", "k", 10),  # duplicate sample, distinct id
            _ev("right", "rp1", "r1", "k", 10),
            _ev("right", "rp1", "r1b", "k", 10),
        ]
        batches = self.run_job(records)
        # l1 pairs with r1 when r1 arrives; r1b then pairs with both lefts.
        self.assertEqual(_rows_of(batches[2]), [("l1", "r1"), ("l1b", "r1")])
        self.assertEqual(
            _rows_of(batches[3]), [("l1", "r1b"), ("l1b", "r1b")]
        )
        # Every pair appears exactly once across the whole log.
        total = sum(len(b["rows"]) for b in batches)
        self.assertEqual(total, 4)

    def test_new_event_immediately_pairs_with_all_received(self) -> None:
        records = [
            _ev("right", "rp1", "r1", "k", 10),
            _ev("right", "rp1", "r2", "k", 12),
            _ev("left", "lp1", "l1", "k", 11),
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[2]), [("l1", "r1"), ("l1", "r2")])

    def test_batch_order_is_ascii_left_then_right(self) -> None:
        records = [
            _ev("right", "rp1", "R1", "k", 10),
            _ev("right", "rp1", "R10", "k", 10),
            _ev("right", "rp1", "R2", "k", 10),
            _ev("left", "lp1", "L9", "k", 10),
            _ev("left", "lp1", "l10", "k", 10),
        ]
        batches = self.run_job(records)
        self.assertEqual(
            _rows_of(batches[3]), [("L9", "R1"), ("L9", "R10"), ("L9", "R2")]
        )
        self.assertEqual(
            _rows_of(batches[4]),
            [("l10", "R1"), ("l10", "R10"), ("l10", "R2")],
        )


class TestWatermarkSemantics(JobCase):
    def test_effective_watermark_is_min_over_all_partitions(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _wm("right", "rp1", 50),  # rp2/rp3 still at -1
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[-1]), [])  # rp1 alone cannot decide l1
        job = JoinJob(self.db)
        self.assertEqual(job.state_summary()["effective_watermarks"]["right"], -1)
        job.close()

    def test_unmatched_left_output_when_right_watermark_reaches_t_plus_5(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _wm("right", "rp1", 15),
            _wm("right", "rp2", 15),
            _wm("right", "rp3", 15),  # effective right = 15 >= 10 + 5
        ]
        batches = self.run_job(records)
        for b in batches[:3]:
            self.assertEqual(_rows_of(b), [])
        self.assertEqual(_rows_of(batches[3]), [("l1", None)])

    def test_no_null_verdict_one_second_before(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _wm("right", "rp1", 14),
            _wm("right", "rp2", 14),
            _wm("right", "rp3", 14),
            _ev("right", "rp3", "r1", "k", 15),  # exactly +5 from l1
        ]
        batches = self.run_job(records)
        self.assertEqual([_rows_of(b) for b in batches[:4]], [[], [], [], []])
        self.assertEqual(_rows_of(batches[4]), [("l1", "r1")])

    def test_matched_left_never_gets_null(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("right", "rp1", "r1", "k", 12),
            _wm("right", "rp1", 30),
            _wm("right", "rp2", 30),
            _wm("right", "rp3", 30),
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[1]), [("l1", "r1")])
        for b in batches[2:]:
            self.assertEqual(_rows_of(b), [])

    def test_unmatched_right_is_never_output(self) -> None:
        records = [
            _ev("right", "rp1", "r1", "k", 10),
            _wm("left", "lp1", 30),
            _wm("left", "lp2", 30),
            _wm("left", "lp3", 30),
        ]
        batches = self.run_job(records)
        for b in batches:
            self.assertEqual(_rows_of(b), [])

    def test_slow_partition_blocks_null_verdict(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _wm("right", "rp1", 100),
            _wm("right", "rp2", 100),
            # rp3 deliberately left at -1: it could still deliver a late match
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[1]), [])
        self.assertEqual(_rows_of(batches[2]), [])

    def test_slow_partition_blocks_right_event_retirement(self) -> None:
        records = [
            _ev("right", "rp1", "r1", "k", 10),
            _wm("left", "lp1", 100),
            _wm("left", "lp2", 100),
            _ev("left", "lp3", "l1", "k", 10),  # lp3 at -1, still in window
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[3]), [("l1", "r1")])
        job = JoinJob(self.db)
        summary = job.state_summary()
        job.close()
        # lp3 is still at -1, so a future left event there could arrive with
        # t = 0 and pair with r1: r1 must stay retained.
        self.assertEqual(summary["right_events_retained"], 1)

    def test_partition_late_events_are_still_joined(self) -> None:
        """A lagging partition's later events can still pair."""
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("left", "lp2", "l2", "k", 155),
            _wm("right", "rp1", 200),
            _wm("right", "rp2", 200),
            # rp3 is still at -1, so effective right watermark stays -1 and
            # no null verdict may be issued yet.
            _ev("right", "rp3", "r3", "k", 150),  # late, but still in window
            _wm("right", "rp3", 200),             # lagging partition catches up
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[4]), [("l2", "r3")])
        self.assertEqual(_rows_of(batches[5]), [("l1", None)])

    def test_watermark_equal_is_legal_but_decrease_is_not(self) -> None:
        records = [
            _wm("left", "lp1", 5),
            _wm("left", "lp1", 5),
            _wm("left", "lp1", 4),
        ]
        job = JoinJob.initialize(self.db, LEFT, RIGHT)
        job.close()
        write_log(self.log, records)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        self.assertEqual(result.batches, 2)
        self.assertEqual(result.stopped_seq, 3)
        job.close()
        self.assertIsNotNone(result.invalid)
        self.assertIsNotNone(result.invalid)
        self.assertEqual(result.stopped_seq, 3)


class TestBatchLog(JobCase):
    def test_empty_batch_is_persisted(self) -> None:
        records = [_wm("left", "lp1", 3), _ev("left", "lp1", "l1", "k", 4)]
        batches = self.run_job(records)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0]["rows"], [])
        self.assertEqual([b["seq"] for b in batches], [1, 2])

    def test_rerun_does_not_add_batches(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("right", "rp1", "r1", "k", 12),
        ]
        job = self.start(records)
        lines_after_first = _read_lines(self.out)
        job.close()
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        self.assertEqual(result.batches, 0)
        job.close()
        self.assertEqual(_read_lines(self.out), lines_after_first)
        self.assertEqual(len(lines_after_first), 2)

    def test_append_continuation_keeps_log_prefix(self) -> None:
        first = [
            _ev("left", "lp1", "l1", "k", 10),
            _wm("right", "rp1", 3),
        ]
        job = self.start(first)
        prefix = _read_text(self.out)
        job.close()

        appended = first + [
            _ev("right", "rp1", "r1", "k", 12),
            _wm("right", "rp2", 20),
            _wm("right", "rp3", 20),
            _wm("right", "rp1", 20),
        ]
        write_log(self.log, appended)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        job.close()
        self.assertEqual(result.batches, 4)
        text = _read_text(self.out)
        self.assertTrue(text.startswith(prefix))
        batches = _read_batches(self.out)
        self.assertEqual(len(batches), 6)
        model, error = run_model_bytes(encode_log(appended), LEFT, RIGHT)
        self.assertIsNone(error)
        self.assertEqual(batches, model)

    def test_eof_does_not_advance_watermarks(self) -> None:
        records = [_ev("left", "lp1", "l1", "k", 10)]
        job = self.start(records)
        batches = job.replay_batches()
        self.assertEqual(_rows_of(batches[-1]), [])
        summary = job.state_summary()
        job.close()
        self.assertEqual(summary["partition_watermarks"]["right"], {
            "rp1": -1, "rp2": -1, "rp3": -1
        })
        self.assertEqual(summary["left_events_retained"], 1)

    def test_invalid_operation_stops_at_previous_success(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 5),
            {"seq": 2, "side": "left", "partition": "lp1", "watermark": 6},
            _ev("left", "lp1", "l2", "k", True, seq=3),  # invalid t
            _ev("left", "lp1", "l3", "k", 7, seq=4),
        ]
        job = JoinJob.initialize(self.db, LEFT, RIGHT)
        job.close()
        write_log(self.log, records)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        self.assertIsNotNone(result.invalid)
        self.assertEqual(result.invalid.seq, 3)
        self.assertEqual(result.stopped_seq, 3)
        summary = job.state_summary()
        job.close()
        self.assertEqual(summary["next_seq"], 3)
        batches = _read_batches(self.out)
        self.assertEqual([b["seq"] for b in batches], [1, 2])
        # The bad record did not corrupt state: appending a fixed copy
        # resumes from seq 3.
        fixed = records[:2] + [dict(records[3], seq=3)]
        write_log(self.log, fixed)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        job.close()
        self.assertEqual(result.batches, 1)
        self.assertEqual(result.stopped_seq, 4)

    def test_invalid_middle_record_does_not_persist_later_records(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 5),
            {"side": "left", "partition": "lp1", "id": "bad", "key": "k",
             "t": -1, "value": None},
            _ev("left", "lp1", "l3", "k", 7),
        ]
        job = JoinJob.initialize(self.db, LEFT, RIGHT)
        job.close()
        write_log(self.log, records)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        job.close()
        self.assertEqual(result.batches, 1)
        self.assertEqual(result.stopped_seq, 2)
        batches = _read_batches(self.out)
        self.assertEqual(len(batches), 1)


class TestValidation(JobCase):
    def _reject(self, bad: list[dict[str, Any]]) -> InvalidOperation:
        records = [_wm("left", "lp1", 0), bad[0]]
        job = JoinJob.initialize(self.db, LEFT, RIGHT)
        job.close()
        write_log(self.log, records)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        job.close()
        self.assertEqual(result.batches, 1)
        self.assertIsNotNone(result.invalid)
        return result.invalid

    def test_extreme_int64_timestamps_supported(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 0),
            _ev("right", "rp1", "r1", "k", 0),
            _ev("left", "lp1", "l2", "k", 2**63 - 1),
            _ev("right", "rp1", "r2", "k", 2**63 - 1),
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[1]), [("l1", "r1")])
        self.assertEqual(_rows_of(batches[3]), [("l2", "r2")])
        model, error = run_model_bytes(encode_log(records), LEFT, RIGHT)
        self.assertIsNone(error)
        self.assertEqual(batches, model)

    def test_boolean_integer_rejected(self) -> None:
        exc = self._reject(
            [{"side": "left", "partition": "lp1", "id": "x", "key": "k",
              "t": True, "value": None}]
        )
        self.assertEqual(exc.seq, 2)

    def test_boolean_watermark_rejected(self) -> None:
        exc = self._reject([{"side": "right", "partition": "rp1", "watermark": False}])
        self.assertEqual(exc.seq, 2)

    def test_int64_overflow_rejected(self) -> None:
        exc = self._reject(
            [{"side": "left", "partition": "lp1", "id": "x", "key": "k",
              "t": 2**63, "value": None}]
        )
        self.assertEqual(exc.seq, 2)

    def test_non_ascii_token_rejected(self) -> None:
        exc = self._reject(
            [{"side": "left", "partition": "lp1", "id": "x", "key": "ключ",
              "t": 1, "value": None}]
        )
        self.assertEqual(exc.seq, 2)

    def test_empty_token_rejected(self) -> None:
        exc = self._reject(
            [{"side": "left", "partition": "", "id": "x", "key": "k",
              "t": 1, "value": None}]
        )
        self.assertEqual(exc.seq, 2)

    def test_numeric_value_rejected(self) -> None:
        exc = self._reject(
            [{"side": "left", "partition": "lp1", "id": "x", "key": "k",
              "t": 1, "value": 0}]
        )
        self.assertEqual(exc.seq, 2)

    def test_duplicate_id_rejected(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 1),
            _ev("left", "lp1", "l1", "k", 2),
        ]
        job = JoinJob.initialize(self.db, LEFT, RIGHT)
        job.close()
        write_log(self.log, records)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        job.close()
        self.assertEqual(result.batches, 1)
        self.assertEqual(result.invalid.seq, 2)

    def test_unknown_partition_rejected(self) -> None:
        exc = self._reject(
            [{"side": "left", "partition": "lp9", "id": "x", "key": "k",
              "t": 1, "value": None}]
        )
        self.assertEqual(exc.seq, 2)

    def test_event_at_or_below_watermark_rejected(self) -> None:
        records = [
            _wm("left", "lp1", 9),
            _ev("left", "lp1", "l1", "k", 9),
        ]
        job = JoinJob.initialize(self.db, LEFT, RIGHT)
        job.close()
        write_log(self.log, records)
        job = JoinJob(self.db)
        result = job.process_log(self.log, self.out)
        job.close()
        self.assertEqual(result.batches, 1)
        self.assertIsNotNone(result.invalid)


class TestLateArrivalReverse(JobCase):
    def test_late_left_after_right_watermark_still_pairs(self) -> None:
        """A left event arriving after the right watermark inside the window."""
        records = [
            _wm("right", "rp1", 5),
            _ev("right", "rp1", "r1", "k", 6),   # watermark 5, so t=6 legal
            _ev("left", "lp1", "l1", "k", 3),    # side-left watermark still -1
            _wm("right", "rp1", 11),
            _wm("right", "rp2", 11),
            _wm("right", "rp3", 11),
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[1]), [])  # r1 alone, no left yet
        self.assertEqual(_rows_of(batches[2]), [("l1", "r1")])  # late left
        # l1 has now matched, so the catch-up watermark adds nothing.
        self.assertEqual(_rows_of(batches[5]), [])

    def test_right_within_five_after_left_watermark(self) -> None:
        records = [
            _wm("left", "lp1", 100),
            _wm("left", "lp2", 100),
            _wm("left", "lp3", 100),
            _ev("right", "rp1", "r1", "k", 98),
            _ev("left", "lp1", "l1", "k", 101),
        ]
        batches = self.run_job(records)
        self.assertEqual(_rows_of(batches[4]), [("l1", "r1")])
        self.assertEqual(batches[4]["rows"][0]["right"]["t"], 98)


class TestStateReclamation(JobCase):
    def test_retention_bounded_after_long_stream(self) -> None:
        records = []
        left_wm = 0
        right_wm = 0
        for i in range(200):
            left_t = left_wm + 1
            records.append(_ev("left", "lp1", f"l{i}", "k", left_t))
            left_wm = left_t
            records.append(_wm("right", "rp1", right_wm + 5))
            right_wm += 5
        for i in range(200):
            right_t = right_wm + 1
            records.append(_ev("right", "rp1", f"r{i}", "other", right_t))
            right_wm = right_t
            records.append(_wm("left", "lp1", left_wm + 5))
            left_wm += 5
        batches = self.run_job(records)
        job = JoinJob(self.db)
        summary = job.state_summary()
        job.close()
        model, error = run_model_bytes(encode_log(records), LEFT, RIGHT)
        self.assertIsNone(error)
        self.assertEqual(batches, model)
        # Slow partitions hold both effective watermarks down at -1, so the
        # state must still be bounded by the number of *live* events, which
        # here is everything; assert instead that no duplicates accumulated.
        self.assertLessEqual(summary["left_events_retained"], 200)
        self.assertLessEqual(summary["right_events_retained"], 200)

    def test_expired_state_is_deleted(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("right", "rp1", "r1", "k", 12),
            _wm("right", "rp1", 100),
            _wm("right", "rp2", 100),
            _wm("right", "rp3", 100),
            _wm("left", "lp1", 100),
            _wm("left", "lp2", 100),
            _wm("left", "lp3", 100),
        ]
        batches = self.run_job(records)
        job = JoinJob(self.db)
        summary = job.state_summary()
        job.close()
        self.assertEqual(summary["left_events_retained"], 0)
        self.assertEqual(summary["right_events_retained"], 0)
        model, error = run_model_bytes(encode_log(records), LEFT, RIGHT)
        self.assertEqual(batches, model)

    def test_only_live_events_retained(self) -> None:
        records = [
            _ev("left", "lp1", "l1", "k", 10),     # will expire
            _ev("left", "lp1", "l2", "k", 90),     # stays live
            _wm("right", "rp1", 30),
            _wm("right", "rp2", 30),
            _wm("right", "rp3", 30),
        ]
        batches = self.run_job(records)
        job = JoinJob(self.db)
        summary = job.state_summary()
        job.close()
        self.assertEqual(_rows_of(batches[4]), [("l1", None)])
        self.assertEqual(summary["left_events_retained"], 1)
        job = JoinJob(self.db)
        self.assertEqual(
            [row[0] for row in job.conn.execute("SELECT id FROM left_events")],
            ["l2"],
        )
        self.assertEqual(job.state_summary()["left_events_retained"], 1)
        job.close()
        model, error = run_model_bytes(encode_log(records), LEFT, RIGHT)
        self.assertEqual(batches, model)


class TestDifferential(unittest.TestCase):
    def test_random_logs_match_full_history_model(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-diff-")
        for seed in range(1, 61):
            records = generate_log(seed, 60)
            db = os.path.join(ws, f"{seed}.db")
            log = os.path.join(ws, f"{seed}.jsonl")
            out = os.path.join(ws, f"{seed}.out.jsonl")
            write_log(log, records)
            JoinJob.initialize(db, LEFT_PARTS, RIGHT_PARTS)
            result, summary, batches = run_engine(
                db, log, out, LEFT_PARTS, RIGHT_PARTS
            )
            self.assertIsNone(result.invalid)
            model, error = run_model_bytes(
                encode_log(records), LEFT_PARTS, RIGHT_PARTS
            )
            self.assertIsNone(error)
            self.assertEqual(len(batches), len(model), f"seed {seed}")
            for actual, expected in zip(batches, model):
                self.assertEqual(
                    actual, expected, f"seed {seed} seq {expected['seq']}"
                )
            live_left, live_right = predicted_retention(records)
            self.assertEqual(
                summary["left_events_retained"], live_left, f"seed {seed}"
            )
            self.assertEqual(
                summary["right_events_retained"], live_right, f"seed {seed}"
            )

    def test_batches_are_byte_identical_to_model(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-bytes-")
        for seed in range(1, 21):
            records = generate_log(seed, 40)
            db = os.path.join(ws, f"{seed}.db")
            log = os.path.join(ws, f"{seed}.jsonl")
            out = os.path.join(ws, f"{seed}.out.jsonl")
            write_log(log, records)
            JoinJob.initialize(db, LEFT_PARTS, RIGHT_PARTS)
            run_engine(db, log, out, LEFT_PARTS, RIGHT_PARTS)
            model, error = run_model_bytes(
                encode_log(records), LEFT_PARTS, RIGHT_PARTS
            )
            self.assertIsNone(error)
            expected = "".join(serialize_batch(b) + "\n" for b in model)
            self.assertEqual(_read_text(out), expected, f"seed {seed}")

    def test_edge_clustered_logs_match_model(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-edge-")
        for seed in range(1, 31):
            records = generate_edge_log(seed, 120)
            db = os.path.join(ws, f"{seed}.db")
            log = os.path.join(ws, f"{seed}.jsonl")
            out = os.path.join(ws, f"{seed}.out.jsonl")
            write_log(log, records)
            JoinJob.initialize(db, LEFT_PARTS, RIGHT_PARTS)
            result, summary, batches = run_engine(
                db, log, out, LEFT_PARTS, RIGHT_PARTS
            )
            self.assertIsNone(result.invalid)
            model, error = run_model_bytes(
                encode_log(records), LEFT_PARTS, RIGHT_PARTS
            )
            self.assertIsNone(error)
            self.assertEqual(batches, model, f"seed {seed}")
            live_left, live_right = predicted_retention(records)
            self.assertEqual(
                summary["left_events_retained"], live_left, f"seed {seed}"
            )
            self.assertEqual(
                summary["right_events_retained"], live_right, f"seed {seed}"
            )

    def test_invalid_logs_stop_identically(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-diff-bad-")
        for seed in range(1, 41):
            invalid_at = (seed % 50) + 2
            records = generate_log(seed, 50, invalid_at)
            db = os.path.join(ws, f"{seed}.db")
            log = os.path.join(ws, f"{seed}.jsonl")
            out = os.path.join(ws, f"{seed}.out.jsonl")
            write_log(log, records)
            JoinJob.initialize(db, LEFT_PARTS, RIGHT_PARTS)
            result, _summary, batches = run_engine(
                db, log, out, LEFT_PARTS, RIGHT_PARTS
            )
            model, error = run_model_bytes(
                encode_log(records), LEFT_PARTS, RIGHT_PARTS
            )
            self.assertIsNotNone(result.invalid, f"seed {seed}")
            self.assertEqual(result.invalid.seq, invalid_at, f"seed {seed}")
            self.assertEqual(error.seq, invalid_at, f"seed {seed}")
            self.assertEqual(batches, model, f"seed {seed}")
            prefix_records = records[: invalid_at - 1]
            live_left, live_right = predicted_retention(prefix_records)
            self.assertEqual(
                _summary["left_events_retained"], live_left, f"seed {seed}"
            )
            self.assertEqual(
                _summary["right_events_retained"], live_right, f"seed {seed}"
            )


class TestCrashRecovery(unittest.TestCase):
    """Hard-kill the process around commits and compare to a clean baseline."""

    LOG_OPERATIONS = [
        _ev("left", "lp1", "l1", "k", 10, "a"),
        _ev("left", "lp2", "l2", "k", 11, None),
        _ev("right", "rp1", "r1", "k", 14, "b"),
        _wm("right", "rp1", 16),
        _ev("right", "rp2", "r2", "k", 20, "c"),
        _wm("right", "rp2", 25),
        _wm("right", "rp3", 25),
        _ev("left", "lp3", "l3", "k", 26, "d"),
        _wm("left", "lp1", 40),
        _wm("left", "lp2", 40),
        _wm("left", "lp3", 40),
        _ev("left", "lp1", "l4", "k", 41, "e"),
    ]

    def _baseline(self, ws: str) -> list[str]:
        db = os.path.join(ws, "baseline.db")
        log = os.path.join(ws, "baseline.jsonl")
        out = os.path.join(ws, "baseline.out.jsonl")
        write_log(log, self.LOG_OPERATIONS)
        job = JoinJob.initialize(db, LEFT, RIGHT)
        job.close()
        with JoinJob(db) as job:
            result = job.process_log(log, out)
        self.assertIsNone(result.invalid)
        return _read_lines(out)

    def _crash_run(
        self, ws: str, point: str, seq: int
    ) -> tuple[list[str], dict[str, Any]]:
        db = os.path.join(ws, f"crash-{point}-{seq}.db")
        log = os.path.join(ws, f"crash-{point}-{seq}.jsonl")
        out = os.path.join(ws, f"crash-{point}-{seq}.out.jsonl")
        write_log(log, self.LOG_OPERATIONS)
        job = JoinJob.initialize(db, LEFT, RIGHT)
        job.close()
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, JOINJOB_FAULT=f"{point}:{seq}", PYTHONPATH=repo)
        proc = subprocess.run(
            [sys.executable, "-m", "joinjob", "run",
             "--db", db, "--input", log, "--output", out],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 137, proc.stderr)
        with JoinJob(db) as job:
            result = job.process_log(log, out)
            summary = job.state_summary()
        self.assertIsNone(result.invalid)
        return _read_lines(out), summary

    def test_crash_before_commit_recovers_exactly(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-crash-")
        baseline = self._baseline(ws)
        # Kill at every operation boundary, not just a few samples.
        for seq in range(1, len(self.LOG_OPERATIONS) + 1):
            lines, summary = self._crash_run(ws, "before_commit", seq)
            self.assertEqual(lines, baseline, f"before_commit seq {seq}")
            self.assertEqual(len(lines), len(baseline))
            self.assertEqual(summary["next_seq"], len(self.LOG_OPERATIONS) + 1)

    def test_crash_after_commit_recovers_exactly(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-crash-")
        baseline = self._baseline(ws)
        for seq in range(1, len(self.LOG_OPERATIONS) + 1):
            lines, summary = self._crash_run(ws, "after_commit", seq)
            self.assertEqual(lines, baseline, f"after_commit seq {seq}")
            self.assertEqual(len(lines), len(baseline))
            self.assertEqual(summary["next_seq"], len(self.LOG_OPERATIONS) + 1)

    def test_crash_recovery_reclaims_expired_state(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-crash-")
        baseline = self._baseline(ws)
        lines, summary = self._crash_run(ws, "after_commit", 8)
        self.assertEqual(lines, baseline)
        # Final left watermark is 40 and right is 25: the two early left
        # events are expired and reclaimed, the two still inside the right
        # window legitimately remain.
        self.assertEqual(summary["left_events_retained"], 2)
        self.assertEqual(summary["right_events_retained"], 0)


class TestEofContinuation(unittest.TestCase):
    def test_unterminated_tail_is_left_for_next_run(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-eof-")
        db = os.path.join(ws, "job.db")
        log = os.path.join(ws, "ops.jsonl")
        out = os.path.join(ws, "batches.jsonl")
        first = [_ev("left", "lp1", "l1", "k", 10)]
        write_bytes(log, encode_log(first) + b'{"side":"left"')
        job = JoinJob.initialize(db, LEFT, RIGHT)
        job.close()
        with JoinJob(db) as job:
            result = job.process_log(log, out)
        self.assertTrue(result.unterminated_tail)
        self.assertEqual(result.batches, 1)
        lines = _read_lines(out)
        self.assertEqual(len(lines), 1)

        # The writer finishes the line; a second run consumes it only.
        second = first + [_wm("right", "rp1", 3)]
        write_log(log, second)
        with JoinJob(db) as job:
            result = job.process_log(log, out)
        self.assertEqual(result.batches, 1)
        lines = _read_lines(out)
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["seq"], 1)
        self.assertEqual(json.loads(lines[1])["seq"], 2)

        model, error = run_model_bytes(encode_log(second), LEFT, RIGHT)
        self.assertIsNone(error)
        self.assertEqual([json.loads(l) for l in lines], model)

    def test_pending_events_survive_eof_and_are_paired_later(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-eof-")
        db = os.path.join(ws, "job.db")
        log = os.path.join(ws, "ops.jsonl")
        out = os.path.join(ws, "batches.jsonl")
        first = [
            _ev("left", "lp1", "l1", "k", 10),
            _ev("right", "rp1", "r1", "k", 12),
        ]
        write_log(log, first)
        job = JoinJob.initialize(db, LEFT, RIGHT)
        job.close()
        with JoinJob(db) as job:
            job.process_log(log, out)
            self.assertEqual(job.state_summary()["left_events_retained"], 1)
        # Appending a right event that pairs with the pending left.
        second = first + [_ev("right", "rp2", "r2", "k", 15)]
        write_log(log, second)
        with JoinJob(db) as job:
            result = job.process_log(log, out)
        self.assertEqual(result.batches, 1)
        final = _read_batches(out)
        self.assertEqual(_rows_of(final[2]), [("l1", "r2")])
        self.assertEqual(final[2]["rows"][0]["left"]["t"], 10)
        model, error = run_model_bytes(encode_log(second), LEFT, RIGHT)
        self.assertEqual(final, model)

    def test_append_with_writer_advancing_in_chunks(self) -> None:
        ws = tempfile.mkdtemp(prefix="joinjob-eof-")
        db = os.path.join(ws, "job.db")
        log = os.path.join(ws, "ops.jsonl")
        out = os.path.join(ws, "batches.jsonl")
        records = generate_log(99, 40)
        job = JoinJob.initialize(db, LEFT, RIGHT)
        job.close()
        data = b""
        with JoinJob(db) as job, open(log, "wb") as writer:
            for i, record in enumerate(records):
                chunk = encode_log([record])
                writer.write(chunk)
                writer.flush()
                data += chunk
                job.process_log(log, out)
            summary = job.state_summary()
        batches = _read_batches(out)
        model, error = run_model_bytes(data, LEFT, RIGHT)
        self.assertIsNone(error)
        self.assertEqual(batches, model)
        self.assertEqual(summary["next_seq"], len(records) + 1)


if __name__ == "__main__":
    unittest.main()
