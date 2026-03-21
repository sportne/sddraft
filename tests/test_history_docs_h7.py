"""Tests for history-docs H7 dependency inventory and summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from engllm.cli.main import main
from engllm.core.config.loader import load_project_config
from engllm.core.repo.history import read_file_at_commit
from engllm.domain.errors import LLMError
from engllm.llm.base import StructuredGenerationRequest
from engllm.llm.mock import MockLLMClient
from engllm.tools.history_docs import dependencies as dependency_module
from engllm.tools.history_docs.build import build_history_docs_checkpoint
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryDependencyConcept,
    HistoryDependencyEntry,
    HistoryDependencyInventory,
    HistoryEvidenceLink,
    HistoryModuleConcept,
)
from tests.history_docs_helpers import (
    checkpoint_model_path,
    commit_file,
    dependencies_artifact_path,
    init_repo,
    write_project_config,
)


class _FailingDependencyClient:
    def generate_structured(self, request: StructuredGenerationRequest):
        raise LLMError(f"boom for {request.model_name}")


def _module(
    path: str,
    *,
    imports: list[str],
    subsystem_id: str | None = None,
    language: str = "python",
) -> HistoryModuleConcept:
    return HistoryModuleConcept(
        concept_id=f"module::{path}",
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        path=Path(path),
        subsystem_id=subsystem_id,
        language=language,
        functions=[],
        classes=[],
        imports=imports,
        docstrings=[],
        symbol_names=[],
        evidence_links=[HistoryEvidenceLink(kind="file", reference=path)],
    )


def _dependency_concept(
    path: str,
    *,
    ecosystem: str,
    related_subsystem_ids: list[str] | None = None,
) -> HistoryDependencyConcept:
    return HistoryDependencyConcept(
        concept_id=f"dependency-source::{path}",
        lifecycle_status="active",
        change_status="observed",
        first_seen_checkpoint="2024-01-01-abc1234",
        last_updated_checkpoint="2024-01-01-abc1234",
        path=Path(path),
        ecosystem=ecosystem,
        category="dependency_manifest",
        related_subsystem_ids=related_subsystem_ids or [],
        evidence_links=[HistoryEvidenceLink(kind="build_source", reference=path)],
    )


@pytest.mark.parametrize(
    ("path", "ecosystem", "content", "expected"),
    [
        (
            Path("pyproject.toml"),
            "python",
            """
[project]
dependencies = ["requests>=2", "pydantic>=2"]
[project.optional-dependencies]
dev = ["pytest>=8"]
""",
            [
                ("pydantic", "runtime"),
                ("pytest", "development"),
                ("requests", "runtime"),
            ],
        ),
        (
            Path("package.json"),
            "javascript",
            '{"dependencies":{"react":"^18.0.0"},"devDependencies":{"eslint":"^9.0.0"}}',
            [("eslint", "development"), ("react", "runtime")],
        ),
        (
            Path("package-lock.json"),
            "javascript",
            '{"packages":{"":{"dependencies":{"react":"^18.0.0"},"devDependencies":{"vitest":"^1.0.0"}}}}',
            [("react", "runtime"), ("vitest", "development")],
        ),
        (
            Path("Cargo.toml"),
            "rust",
            """
[dependencies]
serde = "1"
[dev-dependencies]
insta = "1"
[build-dependencies]
cc = "1"
""",
            [("cc", "build"), ("insta", "test"), ("serde", "runtime")],
        ),
        (
            Path("go.mod"),
            "go",
            """
module example.com/demo

require (
    github.com/spf13/cobra v1.8.0
    golang.org/x/text v0.14.0 // indirect
)
""",
            [("github.com/spf13/cobra", "runtime")],
        ),
        (
            Path("requirements-dev.txt"),
            "python",
            "pytest>=8\nmypy==1.11\n",
            [("mypy", "development"), ("pytest", "development")],
        ),
        (
            Path("Pipfile"),
            "python",
            """
[packages]
requests = "*"
[dev-packages]
pytest = "*"
""",
            [("pytest", "development"), ("requests", "runtime")],
        ),
        (
            Path("Pipfile.lock"),
            "python",
            '{"default":{"requests":{"version":"==2.32.0"}},"develop":{"pytest":{"version":"==8.0.0"}}}',
            [("pytest", "development"), ("requests", "runtime")],
        ),
        (
            Path("pom.xml"),
            "jvm",
            """
<project>
  <dependencies>
    <dependency>
      <groupId>org.slf4j</groupId>
      <artifactId>slf4j-api</artifactId>
      <version>2.0.0</version>
    </dependency>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.13.2</version>
      <scope>test</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-compiler-plugin</artifactId>
      </plugin>
    </plugins>
  </build>
</project>
""",
            [
                ("junit:junit", "test"),
                ("org.apache.maven.plugins:maven-compiler-plugin", "plugin"),
                ("org.slf4j:slf4j-api", "runtime"),
            ],
        ),
        (
            Path("build.gradle.kts"),
            "jvm",
            """
plugins { kotlin("jvm") version "1.9.0" }
dependencies {
  implementation("org.slf4j:slf4j-api:2.0.0")
  testImplementation("junit:junit:4.13.2")
  kapt("com.google.dagger:dagger-compiler:2.0")
}
""",
            [
                ("com.google.dagger:dagger-compiler", "build"),
                ("junit:junit", "test"),
                ("org.slf4j:slf4j-api", "runtime"),
            ],
        ),
        (
            Path("src/App.csproj"),
            "dotnet",
            """
<Project>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="coverlet.collector" Version="6.0.0" PrivateAssets="all" />
  </ItemGroup>
</Project>
""",
            [("coverlet.collector", "build"), ("newtonsoft.json", "unknown")],
        ),
        (
            Path("packages.lock.json"),
            "dotnet",
            '{"dependencies":{"Newtonsoft.Json":{"type":"Direct","resolved":"13.0.3"},"SkipMe":{"type":"Transitive"}}}',
            [("newtonsoft.json", "unknown")],
        ),
        (
            Path("vcpkg.json"),
            "cpp",
            '{"dependencies": ["fmt", {"name": "zlib"}]}',
            [("fmt", "runtime"), ("zlib", "runtime")],
        ),
        (
            Path("conanfile.txt"),
            "cpp",
            """
[requires]
openssl/3.0.0
[tool_requires]
cmake/3.28.0
""",
            [("cmake", "toolchain"), ("openssl", "runtime")],
        ),
        (
            Path("conanfile.py"),
            "cpp",
            'requires = ["openssl/3.0.0"]\ntool_requires = ["cmake/3.28.0"]\n',
            [("cmake", "toolchain"), ("openssl", "runtime")],
        ),
        (
            Path("meson.build"),
            "cpp",
            "project('demo')\nlibarchive_dep = dependency('libarchive')\n",
            [("libarchive", "runtime")],
        ),
        (
            Path("CMakeLists.txt"),
            "cpp",
            "find_package(OpenSSL REQUIRED)\n",
            [("openssl", "build")],
        ),
        (
            Path("WORKSPACE"),
            "generic",
            'http_archive(name = "rules_python")\n',
            [("rules_python", "build")],
        ),
    ],
)
def test_parse_source_covers_representative_manifest_fixtures(
    path: Path,
    ecosystem: str,
    content: str,
    expected: list[tuple[str, str]],
) -> None:
    declarations, warnings = dependency_module._parse_source(
        source_path=path,
        ecosystem=ecosystem,
        content=content,
    )

    observed = sorted((item.normalized_name, item.role) for item in declarations)
    assert observed == expected
    assert all(warning.code != "metadata_only_skipped" for warning in warnings)


def test_parse_source_records_dynamic_and_metadata_warnings_without_fabrication() -> (
    None
):
    declarations, warnings = dependency_module._parse_source(
        source_path=Path("build.gradle.kts"),
        ecosystem="jvm",
        content="dependencies { implementation(libs.slf4j) }\n",
    )
    assert declarations == []
    assert [warning.code for warning in warnings] == ["gradle_dynamic_skipped"]

    declarations, warnings = dependency_module._parse_source(
        source_path=Path("setup.py"),
        ecosystem="python",
        content='setup(name="demo")\n',
    )
    assert declarations == []
    assert [warning.code for warning in warnings] == ["metadata_only_skipped"]


def test_dependency_inventory_path_is_deterministic() -> None:
    tool_root = Path("artifacts/workspaces/demo/tools/history_docs")

    assert dependency_module.dependency_inventory_path(
        tool_root,
        "2024-01-01-abc1234",
    ) == (tool_root / "checkpoints" / "2024-01-01-abc1234" / "dependencies.json")


def test_build_dependency_inventory_aggregates_lockfile_with_primary_and_links_usage(
    tmp_path: Path,
) -> None:
    repo_root = init_repo(tmp_path)
    commit_file(
        repo_root,
        "package.json",
        '{"dependencies":{"react":"^18.0.0"}}\n',
        message="add package manifest",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "package-lock.json",
        '{"packages":{"":{"dependencies":{"react":"^18.0.0"}}}}\n',
        message="add package lock",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-01-02-abc1234",
        target_commit=head,
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        modules=[
            _module("src/web.ts", imports=["react", "./local"], language="typescript")
        ],
        dependencies=[
            _dependency_concept("package.json", ecosystem="javascript"),
            _dependency_concept("package-lock.json", ecosystem="javascript"),
        ],
        subsystems=[],
        sections=[],
    )

    inventory = dependency_module.build_dependency_inventory(
        repo_root=repo_root,
        checkpoint_id="2024-01-02-abc1234",
        target_commit=head,
        previous_checkpoint_commit=None,
        checkpoint_model=checkpoint_model,
        llm_client=MockLLMClient(),
        model_name="mock-engllm",
        temperature=0.2,
        read_file_at_commit=read_file_at_commit,
    )

    assert [entry.dependency_id for entry in inventory.entries] == [
        "dependency::javascript::react"
    ]
    entry = inventory.entries[0]
    assert len(entry.declarations) == 2
    assert entry.related_module_ids == ["module::src/web.ts"]
    assert entry.usage_signals == ["src/web.ts imports react"]


def test_import_usage_linking_is_limited_to_supported_ecosystems() -> None:
    python_entry = HistoryDependencyEntry(
        dependency_id="dependency::python::requests",
        display_name="requests",
        normalized_name="requests",
        ecosystem="python",
        section_target="dependencies",
    )
    jvm_entry = HistoryDependencyEntry(
        dependency_id="dependency::jvm::org.slf4j:slf4j-api",
        display_name="org.slf4j:slf4j-api",
        normalized_name="org.slf4j:slf4j-api",
        ecosystem="jvm",
        section_target="dependencies",
    )
    module = _module(
        "src/app.py",
        imports=["requests", "org.slf4j"],
        subsystem_id="subsystem::src::app",
    )

    related_module_ids, _, _ = dependency_module._related_usage(python_entry, [module])
    assert related_module_ids == ["module::src/app.py"]

    related_module_ids, _, _ = dependency_module._related_usage(jvm_entry, [module])
    assert related_module_ids == []


def test_build_dependency_inventory_marks_lockfile_only_entries_as_tbd(
    tmp_path: Path,
) -> None:
    repo_root = init_repo(tmp_path)
    head = commit_file(
        repo_root,
        "package-lock.json",
        '{"packages":{"":{"dependencies":{"react":"^18.0.0"}}}}\n',
        message="add only package lock",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit=head,
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        modules=[],
        dependencies=[_dependency_concept("package-lock.json", ecosystem="javascript")],
        subsystems=[],
        sections=[],
    )

    inventory = dependency_module.build_dependency_inventory(
        repo_root=repo_root,
        checkpoint_id="2024-01-01-abc1234",
        target_commit=head,
        previous_checkpoint_commit=None,
        checkpoint_model=checkpoint_model,
        llm_client=MockLLMClient(),
        model_name="mock-engllm",
        temperature=0.2,
        read_file_at_commit=read_file_at_commit,
    )

    assert inventory.entries[0].summary_status == "tbd"
    assert inventory.entries[0].general_description == "TBD"
    assert inventory.warnings[-1].code == "lockfile_only_summary_skipped"


def test_link_dependency_inventory_to_checkpoint_model_updates_counts() -> None:
    checkpoint_model = HistoryCheckpointModel(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        previous_checkpoint_commit=None,
        previous_checkpoint_model_available=False,
        modules=[],
        dependencies=[
            _dependency_concept("package.json", ecosystem="javascript"),
            _dependency_concept("pyproject.toml", ecosystem="python"),
        ],
        subsystems=[],
        sections=[],
    )
    inventory = HistoryDependencyInventory(
        checkpoint_id="2024-01-01-abc1234",
        target_commit="a" * 40,
        entries=[
            HistoryDependencyEntry(
                dependency_id="dependency::javascript::react",
                display_name="react",
                normalized_name="react",
                ecosystem="javascript",
                source_dependency_concept_ids=["dependency-source::package.json"],
                section_target="dependencies",
            ),
            HistoryDependencyEntry(
                dependency_id="dependency::python::requests",
                display_name="requests",
                normalized_name="requests",
                ecosystem="python",
                source_dependency_concept_ids=["dependency-source::pyproject.toml"],
                section_target="dependencies",
            ),
        ],
    )

    linked = dependency_module.link_dependency_inventory_to_checkpoint_model(
        checkpoint_model,
        inventory,
    )

    assert linked.dependencies[0].documented_dependency_ids == [
        "dependency::javascript::react"
    ]
    assert linked.dependencies[0].documented_dependency_count == 1
    assert linked.dependencies[1].documented_dependency_ids == [
        "dependency::python::requests"
    ]


def test_build_history_docs_checkpoint_h7_writes_dependencies_artifact_and_links_checkpoint_model(
    tmp_path: Path,
) -> None:
    repo_root = init_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    commit_file(
        repo_root,
        "src/app.py",
        "import requests\n",
        message="add python app",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = ["requests>=2"]
[project.optional-dependencies]
dev = ["pytest>=8"]
""",
        message="add python deps",
        timestamp="2024-01-02T10:00:00+00:00",
    )
    commit_file(
        repo_root,
        "src/web.ts",
        "import React from 'react'\n",
        message="add web module",
        timestamp="2024-01-03T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "package.json",
        '{"dependencies":{"react":"^18.0.0"},"devDependencies":{"eslint":"^9.0.0"}}\n',
        message="add js deps",
        timestamp="2024-01-04T10:00:00+00:00",
    )

    result = build_history_docs_checkpoint(
        project_config=load_project_config(config_path),
        repo_root=repo_root,
        checkpoint_commit=head,
    )

    assert result.dependencies_artifact_path is not None
    assert result.documented_dependency_count == 4
    inventory = HistoryDependencyInventory.model_validate_json(
        result.dependencies_artifact_path.read_text(encoding="utf-8")
    )
    assert [entry.dependency_id for entry in inventory.entries] == [
        "dependency::javascript::eslint",
        "dependency::javascript::react",
        "dependency::python::pytest",
        "dependency::python::requests",
    ]
    assert inventory.entries[1].related_module_ids == ["module::src/web.ts"]

    checkpoint_model = HistoryCheckpointModel.model_validate_json(
        checkpoint_model_path(
            output_root,
            repo_root.name,
            result.checkpoint_id,
        ).read_text(encoding="utf-8")
    )
    dependency_links = {
        concept.path.as_posix(): concept.documented_dependency_ids
        for concept in checkpoint_model.dependencies
        if concept.lifecycle_status == "active"
    }
    assert dependency_links["package.json"] == [
        "dependency::javascript::eslint",
        "dependency::javascript::react",
    ]
    assert dependency_links["pyproject.toml"] == [
        "dependency::python::pytest",
        "dependency::python::requests",
    ]

    expected_path = dependencies_artifact_path(
        output_root,
        repo_root.name,
        result.checkpoint_id,
    )
    assert result.dependencies_artifact_path == expected_path


def test_build_history_docs_checkpoint_h7_falls_back_to_tbd_on_llm_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = init_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    commit_file(
        repo_root,
        "src/app.py",
        "import requests\n",
        message="add python app",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = ["requests>=2"]
""",
        message="add python deps",
        timestamp="2024-01-02T10:00:00+00:00",
    )

    monkeypatch.setattr(
        "engllm.tools.history_docs.build.create_llm_client",
        lambda config: _FailingDependencyClient(),
    )

    result = build_history_docs_checkpoint(
        project_config=load_project_config(config_path),
        repo_root=repo_root,
        checkpoint_commit=head,
    )

    assert result.dependency_summary_failure_count == 1
    inventory = HistoryDependencyInventory.model_validate_json(
        result.dependencies_artifact_path.read_text(encoding="utf-8")
    )
    assert inventory.entries[0].summary_status == "llm_failed"
    assert inventory.entries[0].general_description == "TBD"
    assert inventory.entries[0].project_usage_description == "TBD"


def test_history_docs_build_cli_prints_dependencies_artifact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = init_repo(tmp_path)
    output_root = tmp_path / "artifacts"
    config_path = tmp_path / "project.yaml"
    write_project_config(config_path, output_root, source_roots=["repo/src"])

    commit_file(
        repo_root,
        "src/app.py",
        "import requests\n",
        message="add python app",
        timestamp="2024-01-01T10:00:00+00:00",
    )
    head = commit_file(
        repo_root,
        "pyproject.toml",
        """
[project]
dependencies = ["requests>=2"]
""",
        message="add python deps",
        timestamp="2024-01-02T10:00:00+00:00",
    )

    exit_code = main(
        [
            "history-docs",
            "build",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--checkpoint-commit",
            head,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dependencies:" in captured.out
