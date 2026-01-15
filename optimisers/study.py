from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

import json
import time


TrialStatus = Literal["success", "failed"]


@dataclass(frozen=True)
class TrialRecord:
    trial_id: int
    status: TrialStatus
    params: Dict[str, Any]
    metric_value: Optional[float]
    objective_value: Optional[float]
    run_dir: Optional[str]
    notes: str = ""
    error: Optional[str] = None
    duration_s: Optional[float] = None


class StudyStore:
    """
    Append-only JSONL + a convenience JSON snapshot.
    """

    def __init__(self, study_dir: Path):
        self.study_dir = study_dir
        self.history_jsonl = study_dir / "history.jsonl"
        self.history_json = study_dir / "history.json"

    def ensure(self) -> None:
        self.study_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[TrialRecord]:
        if not self.history_jsonl.exists():
            return []
        out: List[TrialRecord] = []
        with self.history_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(
                    TrialRecord(
                        trial_id=int(d["trial_id"]),
                        status=str(d["status"]),  # type: ignore[arg-type]
                        params=dict(d.get("params") or {}),
                        metric_value=None if d.get("metric_value") is None else float(d["metric_value"]),
                        objective_value=None if d.get("objective_value") is None else float(d["objective_value"]),
                        run_dir=d.get("run_dir"),
                        notes=str(d.get("notes") or ""),
                        error=d.get("error"),
                        duration_s=None if d.get("duration_s") is None else float(d["duration_s"]),
                    )
                )
        return out

    def append(self, rec: TrialRecord) -> None:
        self.ensure()
        with self.history_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
        # Maintain a compact snapshot for easy inspection.
        trials = self.load()
        payload = {
            "updated_unix_s": time.time(),
            "trial_count": len(trials),
            "best": _best_trial_payload(trials),
            "trials": [asdict(t) for t in trials],
        }
        with self.history_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def _best_trial_payload(trials: Sequence[TrialRecord]) -> Optional[Mapping[str, Any]]:
    xs = [t for t in trials if t.status == "success" and t.objective_value is not None]
    if not xs:
        return None
    best = max(xs, key=lambda t: float(t.objective_value))
    return asdict(best)

