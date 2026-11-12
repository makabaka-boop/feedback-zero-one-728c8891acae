"""Random operation-log generator and differential harness.

The generator emits logs that obey the input contract (contiguous seq, unique
ids, event ``t`` strictly above its partition watermark, non-decreasing
watermarks) and can optionally splice in exactly one illegal operation while
continuing to write further records afterwards.  Both the durable engine and
the full-history reference model must stop *before* that operation.

The harness runs the engine over the log and compares every committed batch to
the reference model, structurally and byte for byte.
"""

from __future__ import annotations

import json
import random
import tempfile
from typing import Any, Optional

from joinjob.engine import JoinJob

LEFT_PARTS = ["lp1", "lp2", "lp3"]
RIGHT_PARTS = ["rp1", "rp2", "rp3"]
KEYS = ["alpha", "beta", "gamma"]


class LogBuilder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.wm: dict[tuple[str, str], int] = {
            ("left", p): -1 for p in LEFT_PARTS
        }
        self.wm.update({("right", p): -1 for p in RIGHT_PARTS})
        self.records: list[dict[str, Any]] = []
        self._id_seq = 0

    def add(self, record: dict[str, Any]) -> None:
        record.setdefault("seq", len(self.records) + 1)
        self.records.append(record)

    def watermark(
        self, side: Optional[str] = None, partition: Optional[str] = None
    ) -> None:
        side = side or self.rng.choice(["left", "right"])
        parts = LEFT_PARTS if side == "left" else RIGHT_PARTS
        partition = partition or self.rng.choice(parts)
        key = (side, partition)
        if self.rng.random() < 0.15:
            step = 0  # equal watermark is legal (non-decreasing only)
        else:
            step = self.rng.randint(1, 12)
        self.wm[key] += step
        self.add({"side": side, "partition": partition, "watermark": self.wm[key]})

    def event(
        self, side: Optional[str] = None, partition: Optional[str] = None
    ) -> None:
        side = side or self.rng.choice(["left", "right"])
        parts = LEFT_PARTS if side == "left" else RIGHT_PARTS
        partition = partition or self.rng.choice(parts)
        key = (side, partition)
        t = self.wm[key] + self.rng.randint(1, 15)
        value = None if self.rng.random() < 0.2 else f"v{self.rng.randint(0, 9)}"
        self._id_seq += 1
        self.add(
            {
                "side": side,
                "partition": partition,
                "id": f"e{self._id_seq}",
                "key": self.rng.choice(KEYS),
                "t": t,
                "value": value,
            }
        )


def _invalid_record(builder: LogBuilder, rng: random.Random) -> dict[str, Any]:
    seq = len(builder.records) + 1
    kind = rng.choice(
        [
            "bad_json",
            "bad_utf8",
            "bool_t",
            "negative_t",
            "t_equal_watermark",
            "decrease_watermark",
            "unknown_partition",
            "bad_token",
            "bad_value",
            "missing_field",
            "extra_field",
            "seq_jump",
            "bool_watermark",
        ]
    )
    if kind == "bad_json":
        return {"__raw__": b"{not json\n"}
    if kind == "bad_utf8":
        return {
            "__raw__": (
                b'{"side":"left","partition":"lp1","id":"x","key":"k",'
                b'"t":1,"value":"\xff"}\n'
            )
        }
    if kind == "seq_jump":
        return {
            "seq": seq + 5,
            "side": "left",
            "partition": "lp1",
            "id": "x",
            "key": "k",
            "t": 1,
            "value": None,
        }
    side, part = rng.choice(list(builder.wm))
    key = (side, part)
    if kind == "decrease_watermark":
        return {
            "seq": seq,
            "side": side,
            "partition": part,
            "watermark": builder.wm[key] - 1,
        }
    if kind == "unknown_partition":
        return {"seq": seq, "side": "left", "partition": "nope", "watermark": 5}
    if kind == "bad_token":
        return {"seq": seq, "side": "left", "partition": "lp 1", "watermark": 5}
    if kind == "bool_watermark":
        return {"seq": seq, "side": "left", "partition": "lp1", "watermark": True}
    if kind == "missing_field":
        return {
            "seq": seq,
            "side": "left",
            "partition": "lp1",
            "id": "x",
            "key": "k",
            "t": 1,
        }
    if kind == "extra_field":
        return {
            "seq": seq,
            "side": "left",
            "partition": "lp1",
            "id": "x",
            "key": "k",
            "t": 1,
            "value": None,
            "extra": 1,
        }
    if kind == "bool_t":
        return {
            "seq": seq,
            "side": side,
            "partition": part,
            "id": "x",
            "key": "k",
            "t": True,
            "value": None,
        }
    if kind == "negative_t":
        return {
            "seq": seq,
            "side": side,
            "partition": part,
            "id": "x",
            "key": "k",
            "t": -1 * rng.randint(1, 5),
            "value": None,
        }
    if kind == "t_equal_watermark":
        return {
            "seq": seq,
            "side": side,
            "partition": part,
            "id": "x",
            "key": "k",
            "t": builder.wm[key],
            "value": None,
        }
    if kind == "bad_value":
        return {
            "seq": seq,
            "side": side,
            "partition": part,
            "id": "x",
            "key": "k",
            "t": builder.wm[key] + 1,
            "value": 7,
        }
    raise AssertionError(kind)


def generate_edge_log(seed: int, n_ops: int) -> list[dict[str, Any]]:
    """Timestamps clustered around window edges with many duplicate samples.

    Watermarks creep slowly so that events frequently land exactly at
    ``t +- 5`` boundaries, partitions rarely advance together (so the
    effective watermark is often pinned), and the id/key pools are small,
    producing repeated equal timestamps on both sides.
    """
    rng = random.Random(seed * 1_000_003 + 17)
    builder = LogBuilder(seed)
    builder.rng = rng
    for _ in range(n_ops):
        choose = rng.random()
        if choose < 0.30:
            side = rng.choice(["left", "right"])
            parts = LEFT_PARTS if side == "left" else RIGHT_PARTS
            builder.watermark(side, rng.choice(parts))
        elif choose < 0.85:
            side = rng.choice(["left", "right"])
            parts = LEFT_PARTS if side == "left" else RIGHT_PARTS
            part = rng.choice(parts)
            builder._id_seq += 1
            t = max(builder.wm[(side, part)] + 1, rng.randint(0, 60))
            value = None if rng.random() < 0.25 else f"v{rng.randint(0, 3)}"
            builder.add(
                {
                    "side": side,
                    "partition": part,
                    "id": f"e{builder._id_seq}",
                    "key": rng.choice(KEYS),
                    "t": t,
                    "value": value,
                }
            )
        else:
            # duplicate timestamp on the opposite side, same key
            side = rng.choice(["left", "right"])
            parts = LEFT_PARTS if side == "left" else RIGHT_PARTS
            part = rng.choice(parts)
            builder._id_seq += 1
            t = max(builder.wm[(side, part)] + 1, 10 + 5 * rng.randint(0, 8))
            builder.add(
                {
                    "side": side,
                    "partition": part,
                    "id": f"e{builder._id_seq}",
                    "key": rng.choice(KEYS),
                    "t": t,
                    "value": None,
                }
            )
    return builder.records


def generate_log(
    seed: int, n_ops: int, invalid_at: Optional[int] = None
) -> list[dict[str, Any]]:
    """Generate a random log, optionally splicing one illegal record."""
    builder = LogBuilder(seed)
    while len(builder.records) < n_ops:
        if invalid_at is not None and len(builder.records) + 1 == invalid_at:
            builder.records.append(_invalid_record(builder, builder.rng))
            continue
        if builder.rng.random() < 0.35:
            builder.watermark()
        else:
            builder.event()
    return builder.records


def encode_record(record: dict[str, Any]) -> bytes:
    if "__raw__" in record:
        return record["__raw__"]
    return json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"


def encode_log(records: list[dict[str, Any]]) -> bytes:
    return b"".join(encode_record(r) for r in records)


def write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)


def write_log(path: str, records: list[dict[str, Any]]) -> None:
    write_bytes(path, encode_log(records))


def run_engine(
    db_path: str,
    log_path: str,
    output_path: str,
    left: list[str],
    right: list[str],
    max_ops: Optional[int] = None,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    with JoinJob(db_path) as job:
        result = job.process_log(log_path, output_path, max_ops=max_ops)
        summary = job.state_summary()
        batches = job.replay_batches()
    return result, summary, batches


def output_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def make_workspace() -> str:
    return tempfile.mkdtemp(prefix="joinjob-")
