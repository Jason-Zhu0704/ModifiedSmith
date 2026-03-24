"""Runtime tracking utilities for program/service timing reports."""

from __future__ import annotations

import json
import os
import time

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_CURRENT_TRACKER: ContextVar["RuntimeTracker | None"] = ContextVar(
    "_CURRENT_RUNTIME_TRACKER", default=None
)


@dataclass
class RuntimeEvent:
    """Single timing event emitted by program or service calls."""

    sequence: int
    category: str
    name: str
    started_at_unix_s: float
    ended_at_unix_s: float
    duration_s: float
    pid: int
    metadata: dict[str, Any] | None


class RuntimeTracker:
    """Append-only JSONL event tracker safe for multi-process usage."""

    def __init__(self, events_path: Path):
        self.events_path = Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def track(
        self,
        *,
        category: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        start = time.perf_counter()
        started_unix = time.time()
        try:
            yield
        finally:
            end = time.perf_counter()
            ended_unix = time.time()
            self.record(
                category=category,
                name=name,
                started_at_unix_s=started_unix,
                ended_at_unix_s=ended_unix,
                duration_s=end - start,
                metadata=metadata,
            )

    def record(
        self,
        *,
        category: str,
        name: str,
        started_at_unix_s: float,
        ended_at_unix_s: float,
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "category": category,
            "name": name,
            "started_at_unix_s": started_at_unix_s,
            "ended_at_unix_s": ended_at_unix_s,
            "duration_s": duration_s,
            "pid": os.getpid(),
            "metadata": metadata,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=True) + "\n")


@contextmanager
def activate_runtime_tracker(tracker: RuntimeTracker | None) -> Iterator[None]:
    token = _CURRENT_TRACKER.set(tracker)
    try:
        yield
    finally:
        _CURRENT_TRACKER.reset(token)


def get_runtime_tracker() -> RuntimeTracker | None:
    return _CURRENT_TRACKER.get()


@contextmanager
def track_runtime(
    *,
    category: str,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    tracker = get_runtime_tracker()
    if tracker is None:
        yield
        return
    with tracker.track(category=category, name=name, metadata=metadata):
        yield


def load_runtime_events(events_path: Path) -> list[RuntimeEvent]:
    if not events_path.exists():
        return []

    raw_events: list[dict[str, Any]] = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_events.append(json.loads(line))

    raw_events.sort(key=lambda e: e["started_at_unix_s"])

    events: list[RuntimeEvent] = []
    for idx, event in enumerate(raw_events, start=1):
        events.append(
            RuntimeEvent(
                sequence=idx,
                category=event["category"],
                name=event["name"],
                started_at_unix_s=float(event["started_at_unix_s"]),
                ended_at_unix_s=float(event["ended_at_unix_s"]),
                duration_s=float(event["duration_s"]),
                pid=int(event["pid"]),
                metadata=event.get("metadata"),
            )
        )
    return events


def write_runtime_reports(scene_dir: Path, events_path: Path) -> None:
    events = load_runtime_events(events_path=events_path)

    timeline_json = scene_dir / "runtime_timeline_report.json"
    grouped_json = scene_dir / "runtime_program_report.json"
    timeline_md = scene_dir / "runtime_timeline_report.md"
    grouped_md = scene_dir / "runtime_program_report.md"

    timeline_payload = [
        {
            "sequence": event.sequence,
            "category": event.category,
            "name": event.name,
            "duration_s": round(event.duration_s, 4),
            "started_at": datetime.fromtimestamp(
                event.started_at_unix_s, tz=timezone.utc
            ).isoformat(),
            "ended_at": datetime.fromtimestamp(
                event.ended_at_unix_s, tz=timezone.utc
            ).isoformat(),
            "pid": event.pid,
            "metadata": event.metadata,
        }
        for event in events
    ]

    grouped: dict[str, dict[str, Any]] = {}
    for event in events:
        key = f"{event.category}:{event.name}"
        item = grouped.setdefault(
            key,
            {
                "category": event.category,
                "program": event.name,
                "call_count": 0,
                "durations_s": [],
                "total_duration_s": 0.0,
            },
        )
        item["call_count"] += 1
        item["durations_s"].append(round(event.duration_s, 4))
        item["total_duration_s"] += event.duration_s

    grouped_payload = sorted(
        (
            {
                **item,
                "total_duration_s": round(item["total_duration_s"], 4),
                "avg_duration_s": round(
                    item["total_duration_s"] / item["call_count"], 4
                ),
            }
            for item in grouped.values()
        ),
        key=lambda x: x["total_duration_s"],
        reverse=True,
    )

    with open(timeline_json, "w", encoding="utf-8") as f:
        json.dump(timeline_payload, f, indent=2, ensure_ascii=False)

    with open(grouped_json, "w", encoding="utf-8") as f:
        json.dump(grouped_payload, f, indent=2, ensure_ascii=False)

    with open(timeline_md, "w", encoding="utf-8") as f:
        f.write("# Runtime Timeline Report\n\n")
        f.write("| # | Category | Program/Service | Duration (s) | PID |\n")
        f.write("|---|---|---|---:|---:|\n")
        for event in events:
            f.write(
                f"| {event.sequence} | {event.category} | {event.name} | "
                f"{event.duration_s:.4f} | {event.pid} |\n"
            )

    with open(grouped_md, "w", encoding="utf-8") as f:
        f.write("# Runtime Program Report\n\n")
        f.write("| Category | Program/Service | Calls | Durations (s) | Total (s) | Avg (s) |\n")
        f.write("|---|---|---:|---|---:|---:|\n")
        for item in grouped_payload:
            durations = ", ".join(f"{d:.4f}" for d in item["durations_s"])
            f.write(
                f"| {item['category']} | {item['program']} | {item['call_count']} | "
                f"{durations} | {item['total_duration_s']:.4f} | "
                f"{item['avg_duration_s']:.4f} |\n"
            )
