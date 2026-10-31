"""TRL rubric loading and validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class Criterion(BaseModel):
    """A single criterion in the TRL rubric."""

    id: str = Field(..., description="Unique criterion identifier.")
    text: str = Field(..., description="Criterion text.")
    hints: list[str] = Field(..., min_length=5, max_length=8, description="Search hints.")


class Level(BaseModel):
    """A TRL level and its criteria."""

    level: int = Field(..., ge=1, le=9, description="TRL level number.")
    name: str = Field(..., min_length=1, description="Human-readable level name.")
    criteria: list[Criterion] = Field(..., min_length=1, description="Level criteria.")


class Rubric(BaseModel):
    """Validated nine-level TRL rubric."""

    levels: list[Level] = Field(..., description="TRL levels one through nine.")

    @model_validator(mode="after")
    def validate_rubric(self) -> "Rubric":
        """Ensure all levels and criterion identifiers are unique and complete."""
        if [level.level for level in self.levels] != list(range(1, 10)):
            raise ValueError("rubric must contain levels 1 through 9 in order")
        criterion_ids = [criterion.id for level in self.levels for criterion in level.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("criterion ids must be unique")
        if any(not criterion.text.strip() for level in self.levels for criterion in level.criteria):
            raise ValueError("criterion text must be non-empty")
        return self


RUBRIC_PATH = Path(__file__).parent / "data" / "rubric.yaml"


def load_rubric(path: Path = RUBRIC_PATH) -> Rubric:
    """Load and validate a rubric YAML document."""
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Rubric.model_validate(raw)
