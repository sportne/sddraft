"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.core.config.loader import (
    load_config_bundle,
    load_csc_descriptor,
    load_project_config,
    load_sdd_template,
)
from engllm.domain.errors import ConfigError, ValidationError


def test_load_config_bundle_success(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "templates").mkdir()

    project_path = tmp_path / "project.yaml"
    csc_path = tmp_path / "csc.yaml"
    template_path = tmp_path / "templates" / "sdd.yaml"

    project_path.write_text(
        """
project_name: ExampleProject
workspace:
  output_root: artifacts
sources:
  roots: [src]
  include: ["**/*.py"]
  exclude: ["**/tests/**"]
llm:
  provider: mock
  model_name: mock-engllm
  temperature: 0.2
tools:
  sdd:
    template: templates/sdd.yaml
""".strip(),
        encoding="utf-8",
    )

    csc_path.write_text(
        """
csc_id: NAV_CTRL
title: Navigation Control
purpose: Controls navigation.
source_roots: [src]
key_files: [src/module.py]
provided_interfaces: [NavControlService]
used_interfaces: [PositionProvider]
requirements: [SYS-NAV-001]
""".strip(),
        encoding="utf-8",
    )

    template_path.write_text(
        """
document_type: sdd
sections:
  - id: "1"
    title: "Scope"
    instruction: "Describe scope"
    evidence_kinds: [csc_descriptor]
""".strip(),
        encoding="utf-8",
    )

    bundle = load_config_bundle(project_path, [csc_path])

    assert bundle.project.project_name == "ExampleProject"
    assert bundle.project.workspace.output_root.is_absolute()
    assert bundle.project.sources.roots[0].is_absolute()
    assert bundle.csc_descriptors[0].source_roots[0].is_absolute()
    assert bundle.project.tools.sdd.template.is_absolute()
    assert bundle.template.sections[0].title == "Scope"


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("{not: valid: yaml", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_project_config(bad_path)


def test_missing_required_field_raises_validation_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text("project_name: X", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_project_config(missing_path)


def test_load_individual_template_and_csc(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    csc_path = tmp_path / "csc.yaml"

    template_path.write_text(
        """
document_type: sdd
sections:
  - id: "1"
    title: "Scope"
    instruction: "Describe"
""".strip(),
        encoding="utf-8",
    )
    csc_path.write_text(
        """
csc_id: X
title: T
purpose: P
""".strip(),
        encoding="utf-8",
    )

    template = load_sdd_template(template_path)
    csc = load_csc_descriptor(csc_path)

    assert template.document_type == "sdd"
    assert csc.csc_id == "X"


def test_invalid_generation_tuning_raises_validation_error(tmp_path: Path) -> None:
    project_path = tmp_path / "project.yaml"
    project_path.write_text(
        """
project_name: ExampleProject
workspace:
  output_root: artifacts
sources:
  roots: [src]
  include: ["**/*.py"]
  exclude: []
tools:
  sdd:
    template: templates/sdd.yaml
generation:
  max_files: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="generation options must be positive"):
        load_project_config(project_path)
