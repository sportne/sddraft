"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from engllm.domain.errors import ConfigError, ValidationError
from engllm.domain.models import (
    ConfigBundle,
    CSCDescriptor,
    ProjectConfig,
    SDDTemplate,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML at {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Expected mapping at root of YAML file: {path}")
    return raw


def _normalize_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalize_path_list(values: list[str | Path], base_dir: Path) -> list[Path]:
    return [_normalize_path(value, base_dir) for value in values]


def load_project_config(path: Path) -> ProjectConfig:
    """Load and validate the main project configuration file."""

    raw = _load_yaml(path)
    base_dir = path.parent.resolve()

    sources = raw.get("sources", {})
    if not isinstance(sources, dict):
        raise ConfigError("project config 'sources' must be a mapping")

    roots = sources.get("roots", [])
    if not isinstance(roots, list):
        raise ConfigError("project config 'sources.roots' must be a list")
    sources["roots"] = _normalize_path_list(roots, base_dir)

    workspace = raw.get("workspace", {})
    if not isinstance(workspace, dict):
        raise ConfigError("project config 'workspace' must be a mapping")
    if "output_root" in workspace:
        workspace["output_root"] = _normalize_path(workspace["output_root"], base_dir)
    raw["workspace"] = workspace

    tools = raw.get("tools", {})
    if not isinstance(tools, dict):
        raise ConfigError("project config 'tools' must be a mapping")
    sdd = tools.get("sdd", {})
    if not isinstance(sdd, dict):
        raise ConfigError("project config 'tools.sdd' must be a mapping")
    if "template" in sdd:
        sdd["template"] = _normalize_path(sdd["template"], base_dir)
    tools["sdd"] = sdd
    raw["tools"] = tools

    raw["sources"] = sources

    try:
        return ProjectConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid project config at {path}: {exc}") from exc


def load_csc_descriptor(path: Path) -> CSCDescriptor:
    """Load and validate a CSC descriptor YAML file."""

    raw = _load_yaml(path)
    base_dir = path.parent.resolve()

    for field_name in ("source_roots", "key_files"):
        values = raw.get(field_name, [])
        if values:
            if not isinstance(values, list):
                raise ConfigError(f"'{field_name}' must be a list in {path}")
            raw[field_name] = _normalize_path_list(values, base_dir)

    try:
        return CSCDescriptor.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid CSC descriptor at {path}: {exc}") from exc


def load_sdd_template(path: Path) -> SDDTemplate:
    """Load and validate an SDD template YAML file."""

    raw = _load_yaml(path)
    try:
        return SDDTemplate.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid SDD template at {path}: {exc}") from exc


def load_config_bundle(
    project_config_path: Path,
    csc_paths: list[Path],
    template_path: Path | None = None,
) -> ConfigBundle:
    """Load project config, one-or-more CSC descriptors, and template."""

    project = load_project_config(project_config_path)
    csc_descriptors = [load_csc_descriptor(path) for path in csc_paths]

    chosen_template_path = template_path or project.tools.sdd.template
    template = load_sdd_template(chosen_template_path)

    return ConfigBundle(
        project=project,
        csc_descriptors=csc_descriptors,
        template=template,
    )
