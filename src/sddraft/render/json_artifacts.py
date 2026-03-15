"""JSON artifact renderers."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from sddraft.domain.errors import RenderingError


def write_json_model(path: Path, model: BaseModel) -> Path:
    """Write a pydantic model as formatted JSON."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise RenderingError(f"Failed writing JSON artifact to {path}: {exc}") from exc
    return path
