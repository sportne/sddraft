"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.domain.models import (
    CSCDescriptor,
    GenerationOptions,
    LLMConfig,
    ProjectConfig,
    SDDSectionSpec,
    SDDTemplate,
    SDDToolDefaults,
    SourcesConfig,
    ToolDefaults,
    WorkspaceConfig,
)


@pytest.fixture()
def sample_template() -> SDDTemplate:
    return SDDTemplate(
        document_type="sdd",
        sections=[
            SDDSectionSpec(
                id="1",
                title="Scope",
                instruction="Describe scope.",
                evidence_kinds=["csc_descriptor"],
            ),
            SDDSectionSpec(
                id="2",
                title="Design Overview",
                instruction="Describe overview.",
                evidence_kinds=["code_summary", "dependencies"],
            ),
            SDDSectionSpec(
                id="3",
                title="Interface Design",
                instruction="Describe interfaces.",
                evidence_kinds=["interfaces"],
            ),
            SDDSectionSpec(
                id="4",
                title="Detailed Design",
                instruction="Describe details.",
                evidence_kinds=["commit_impact"],
            ),
        ],
    )


@pytest.fixture()
def sample_csc(tmp_path: Path) -> CSCDescriptor:
    return CSCDescriptor(
        csc_id="NAV_CTRL",
        title="Navigation Control",
        purpose="Controls navigation.",
        source_roots=[tmp_path / "src"],
        key_files=[tmp_path / "src" / "module.py"],
        provided_interfaces=["NavControlService"],
        used_interfaces=["PositionProvider"],
        requirements=["SYS-NAV-001"],
    )


@pytest.fixture()
def sample_project_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        project_name="ExampleProject",
        sources=SourcesConfig(
            roots=[tmp_path / "src"],
            include=[
                "**/*.py",
                "**/*.java",
                "**/*.c",
                "**/*.cc",
                "**/*.cpp",
                "**/*.h",
                "**/*.hpp",
                "**/*.js",
                "**/*.mjs",
                "**/*.cjs",
                "**/*.ts",
                "**/*.tsx",
                "**/*.go",
                "**/*.rs",
                "**/*.cs",
            ],
            exclude=["**/tests/**"],
        ),
        workspace=WorkspaceConfig(output_root=tmp_path / "artifacts"),
        tools=ToolDefaults(
            sdd=SDDToolDefaults(template=tmp_path / "templates" / "sdd_default.yaml")
        ),
        llm=LLMConfig(provider="mock", model_name="mock-engllm", temperature=0.2),
        generation=GenerationOptions(
            max_files=100, code_chunk_lines=20, retrieval_top_k=6
        ),
    )
