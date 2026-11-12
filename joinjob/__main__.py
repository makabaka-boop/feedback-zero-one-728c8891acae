"""Command line entry point.

Usage::

    python -m joinjob init  --db job.db --left-parts p1,p2 --right-parts q1,q2
    python -m joinjob run   --db job.db --input ops.jsonl --output batches.jsonl
    python -m joinjob run   --db job.db --input ops.jsonl --output batches.jsonl --max-ops 100
    python -m joinjob status --db job.db

Exit codes: 0 success / clean stop at EOF, 2 invalid operation, 3 unusable
input, 1 unexpected error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .engine import JoinJob, normalize_partitions
from .model import InvalidOperation


def _split(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(",") if part != ""]


def _cmd_init(args: argparse.Namespace) -> int:
    left = normalize_partitions(_split(args.left_parts), "left")
    right = normalize_partitions(_split(args.right_parts), "right")
    JoinJob.initialize(args.db, left, right).close()
    print(
        json.dumps(
            {"initialized": args.db, "left_partitions": left, "right_partitions": right}
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    with JoinJob(args.db) as job:
        result = job.process_log(args.input, args.output, max_ops=args.max_ops)
        summary = job.state_summary()
    print(
        json.dumps(
            {
                "batches_written": result.batches,
                "next_seq": result.stopped_seq,
                "reached_eof": result.reached_eof,
                "unterminated_tail": result.unterminated_tail,
                "invalid": None
                if result.invalid is None
                else {"seq": result.invalid.seq, "reason": result.invalid.reason},
                "retained_left_events": summary["left_events_retained"],
                "retained_right_events": summary["right_events_retained"],
                "consumed_offset": summary["consumed_offset"],
            }
        )
    )
    if result.invalid is not None:
        return 2
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    with JoinJob(args.db) as job:
        print(json.dumps(job.state_summary(), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="joinjob")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a fresh job database")
    p_init.add_argument("--db", required=True)
    p_init.add_argument("--left-parts", required=True)
    p_init.add_argument("--right-parts", required=True)
    p_init.set_defaults(func=_cmd_init)

    p_run = sub.add_parser("run", help="consume the log and commit batches")
    p_run.add_argument("--db", required=True)
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--output")
    p_run.add_argument("--max-ops", type=int, default=None)
    p_run.set_defaults(func=_cmd_run)

    p_status = sub.add_parser("status", help="print durable state summary")
    p_status.add_argument("--db", required=True)
    p_status.set_defaults(func=_cmd_status)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except InvalidOperation as exc:  # defensive; run reports these itself
        print(f"invalid operation: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
