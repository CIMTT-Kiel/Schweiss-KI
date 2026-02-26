"""
Report-Datenstrukturen für das Preprocessing.

PreprocessingReport wird im WeldVolumeModel gespeichert und dokumentiert
automatisch alle angewendeten Schritte mit Parametern und Metriken.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StepReport:
    """Bericht über einen einzelnen Preprocessing-Schritt."""

    step_name: str
    params: dict[str, Any]
    points_before: int
    points_after: int
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def points_removed(self) -> int:
        return self.points_before - self.points_after

    @property
    def retention_rate(self) -> float:
        """Anteil der behaltenen Punkte (0.0–1.0)."""
        if self.points_before == 0:
            return 1.0
        return self.points_after / self.points_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "params": self.params,
            "points_before": self.points_before,
            "points_after": self.points_after,
            "points_removed": self.points_removed,
            "retention_rate": round(self.retention_rate, 4),
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class PreprocessingReport:
    """
    Vollständiger Bericht über eine Preprocessing-Pipeline-Ausführung.

    Wird im WeldVolumeModel gespeichert und ist serialisierbar (JSON).
    """

    source_type: str  # "ideal" | "real" | "synthetic"
    steps: list[StepReport] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)

    @property
    def points_in(self) -> int:
        return self.steps[0].points_before if self.steps else 0

    @property
    def points_out(self) -> int:
        return self.steps[-1].points_after if self.steps else 0

    @property
    def total_retention_rate(self) -> float:
        if self.points_in == 0:
            return 1.0
        return self.points_out / self.points_in

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "created_at": self.created_at,
            "summary": {
                "points_in": self.points_in,
                "points_out": self.points_out,
                "total_retention_rate": round(self.total_retention_rate, 4),
                "total_duration_ms": round(self.total_duration_ms, 2),
                "n_steps": len(self.steps),
            },
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingReport":
        """Rekonstruiert einen PreprocessingReport aus einem JSON-Dict."""
        steps = [
            StepReport(
                step_name=s["step_name"],
                params=s["params"],
                points_before=s["points_before"],
                points_after=s["points_after"],
                duration_ms=s["duration_ms"],
                timestamp=s["timestamp"],
            )
            for s in data.get("steps", [])
        ]
        return cls(
            source_type=data["source_type"],
            steps=steps,
            created_at=data.get("created_at", datetime.now().isoformat()),
        )

    def __repr__(self) -> str:
        return (
            f"PreprocessingReport("
            f"steps={len(self.steps)}, "
            f"retention={self.total_retention_rate:.1%}, "
            f"duration={self.total_duration_ms:.1f}ms)"
        )