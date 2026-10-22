"""Append-only access log. Records IDs only: no names, transcripts or report text."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("viscan.audit")
_lock = threading.Lock()


class AuditLog:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, clinician_id: str, action: str, outcome: str, **ids: str | None) -> None:
        entry = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "clinician_id": clinician_id,
            "action": action,
            "outcome": outcome,
            **{k: v for k, v in ids.items() if v},
        }
        log.info("audit %s", entry)
        if self.path:
            with _lock, self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
