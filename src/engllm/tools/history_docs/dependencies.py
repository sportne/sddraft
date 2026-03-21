"""H7 dependency inventory builders and checkpoint-model linkers."""

from __future__ import annotations

import json
import re
import tomllib
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import cast

from engllm.domain.errors import EngLLMError, GitError
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_dependency_summary_prompt
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryDependencyConcept,
    HistoryDependencyDeclaration,
    HistoryDependencyEntry,
    HistoryDependencyInventory,
    HistoryDependencyRole,
    HistoryDependencySectionTarget,
    HistoryDependencySourceKind,
    HistoryDependencySummary,
    HistoryDependencyWarning,
    HistoryModuleConcept,
)

_PRIMARY_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "Pipfile",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "vcpkg.json",
    "conanfile.txt",
    "conanfile.py",
    "meson.build",
    "CMakeLists.txt",
    "WORKSPACE",
}
_LOCKFILE_NAMES = {
    "Pipfile.lock",
    "package-lock.json",
    "packages.lock.json",
}
_RELIABLE_USAGE_ECOSYSTEMS = {"python", "javascript", "typescript", "rust", "go"}
_PYTHON_DEV_GROUP_TOKENS = {"dev", "docs", "lint", "format", "type", "typing"}
_PYTHON_TEST_GROUP_TOKENS = {"test", "tests", "qa"}
_GROUP_ROLE_MAP = {
    "dependencies": "runtime",
    "devdependencies": "development",
    "dev-dependencies": "development",
    "dev_dependencies": "development",
    "default": "runtime",
    "develop": "development",
    "optional": "optional",
}
_GRADLE_ROLE_MAP: dict[str, HistoryDependencyRole] = {
    "api": "runtime",
    "implementation": "runtime",
    "compileonly": "build",
    "runtimeonly": "runtime",
    "developmentonly": "development",
    "testimplementation": "test",
    "testruntimeonly": "test",
    "testcompileonly": "test",
    "testfixturesimplementation": "test",
    "annotationprocessor": "build",
    "kapt": "build",
    "ksp": "build",
    "classpath": "plugin",
}
_CONAN_SECTION_ROLE_MAP: dict[str, HistoryDependencyRole] = {
    "requires": "runtime",
    "tool_requires": "toolchain",
    "test_requires": "test",
}
_XML_TEXT_RE = re.compile(r"\s+")
_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
_GO_REQUIRE_BLOCK_RE = re.compile(r"require\s*\((?P<body>.*?)\)", re.DOTALL)
_GO_REQUIRE_LINE_RE = re.compile(r"^\s*(?P<name>\S+)\s+(?P<version>\S+)(?P<rest>.*)$")
_GRADLE_DECLARATION_RE = re.compile(
    r"(?P<config>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|\s)\s*['\"](?P<coords>[^'\"]+)['\"]"
)
_GRADLE_DYNAMIC_RE = re.compile(
    r"(?P<config>[A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(?P<expr>libs\.[^)]+|project\([^)]+\))\s*\)"
)
_MESON_DEP_RE = re.compile(r"dependency\(\s*['\"](?P<name>[^'\"]+)['\"]")
_CMAKE_PACKAGE_RE = re.compile(r"find_package\(\s*(?P<name>[A-Za-z0-9_+.-]+)")
_WORKSPACE_REPO_RE = re.compile(
    r"(?:http_archive|git_repository|local_repository|new_local_repository)\s*\(.*?name\s*=\s*['\"](?P<name>[^'\"]+)['\"]",
    re.DOTALL,
)
_CONAN_ASSIGNMENT_RE = re.compile(
    r"(?P<name>requires|tool_requires|test_requires)\s*=\s*(?P<value>\[[^\]]*\]|\([^\)]*\)|['\"][^'\"]+['\"])",
    re.DOTALL,
)
_QUOTED_ITEM_RE = re.compile(r"['\"]([^'\"]+)['\"]")
_IMPORT_TOKEN_RE = re.compile(r"[@A-Za-z0-9_./:-]+")


def dependency_inventory_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H7 dependency inventory artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "dependencies.json"


def _warning(source_path: Path, code: str, message: str) -> HistoryDependencyWarning:
    return HistoryDependencyWarning(source_path=source_path, code=code, message=message)


def _source_kind(path: Path) -> HistoryDependencySourceKind:
    name = path.name
    if (
        name in _PRIMARY_NAMES
        or name.startswith("requirements-")
        or path.suffix == ".csproj"
    ):
        return "primary"
    if name in _LOCKFILE_NAMES:
        return "lockfile"
    return "metadata"


def _normalize_name(ecosystem: str, name: str) -> str:
    normalized = name.strip()
    if ecosystem != "unknown":
        normalized = normalized.lower()
    if ecosystem in {"python", "rust"}:
        normalized = normalized.replace("_", "-")
    return normalized


def _dependency_id(ecosystem: str, normalized_name: str) -> str:
    return f"dependency::{ecosystem}::{normalized_name}"


def _dependency_concept_id(path: Path) -> str:
    return f"dependency-source::{path.as_posix()}"


def _normalize_text(value: str) -> str:
    return _XML_TEXT_RE.sub(" ", value).strip()


def _normalize_group_name(group_name: str | None) -> str:
    return "" if group_name is None else group_name.lower().replace("_", "-")


def _python_group_role(group_name: str | None) -> HistoryDependencyRole:
    normalized = _normalize_group_name(group_name)
    if not normalized:
        return "runtime"
    if normalized in _PYTHON_TEST_GROUP_TOKENS:
        return "test"
    if normalized in _PYTHON_DEV_GROUP_TOKENS:
        return "development"
    if "test" in normalized:
        return "test"
    if any(token in normalized for token in ("dev", "lint", "docs", "type")):
        return "development"
    return "optional"


def _role_from_group_name(group_name: str | None) -> HistoryDependencyRole:
    normalized = _normalize_group_name(group_name)
    if not normalized:
        return "runtime"
    if normalized in _GROUP_ROLE_MAP:
        return cast(HistoryDependencyRole, _GROUP_ROLE_MAP[normalized])
    return _python_group_role(group_name)


def _parse_requirement_name(raw: str) -> str | None:
    match = _REQUIREMENT_NAME_RE.match(raw)
    if match is None:
        return None
    return match.group(1)


def _declaration(
    *,
    source_path: Path,
    source_kind: HistoryDependencySourceKind,
    ecosystem: str,
    raw_name: str,
    role: HistoryDependencyRole,
    version_spec: str | None = None,
    group_name: str | None = None,
    declaration_text: str | None = None,
) -> HistoryDependencyDeclaration:
    return HistoryDependencyDeclaration(
        source_path=source_path,
        source_kind=source_kind,
        raw_name=raw_name,
        normalized_name=_normalize_name(ecosystem, raw_name),
        role=role,
        version_spec=version_spec,
        group_name=group_name,
        declaration_text=declaration_text,
    )


def _append_mapping_declarations(
    *,
    declarations: list[HistoryDependencyDeclaration],
    source_path: Path,
    source_kind: HistoryDependencySourceKind,
    ecosystem: str,
    mapping: object,
    role: HistoryDependencyRole,
    group_name: str | None = None,
) -> None:
    if not isinstance(mapping, dict):
        return
    for raw_name in sorted(mapping):
        if ecosystem == "python" and raw_name.lower() == "python":
            continue
        raw_spec = mapping[raw_name]
        version_spec = None
        if isinstance(raw_spec, str):
            version_spec = raw_spec
        elif isinstance(raw_spec, dict):
            version_spec = json.dumps(raw_spec, sort_keys=True)
        declarations.append(
            _declaration(
                source_path=source_path,
                source_kind=source_kind,
                ecosystem=ecosystem,
                raw_name=raw_name,
                role=role,
                version_spec=version_spec,
                group_name=group_name,
                declaration_text=version_spec,
            )
        )


def _parse_pyproject(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    warnings: list[HistoryDependencyWarning] = []
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return declarations, [_warning(source_path, "toml_parse_failed", str(exc))]

    project = data.get("project", {})
    if isinstance(project, dict):
        for item in project.get("dependencies", []) or []:
            if not isinstance(item, str):
                continue
            raw_name = _parse_requirement_name(item)
            if raw_name is None:
                warnings.append(
                    _warning(source_path, "python_requirement_unparsed", item)
                )
                continue
            declarations.append(
                _declaration(
                    source_path=source_path,
                    source_kind=source_kind,
                    ecosystem="python",
                    raw_name=raw_name,
                    role="runtime",
                    version_spec=item,
                    declaration_text=item,
                )
            )
        optional_groups = project.get("optional-dependencies", {})
        if isinstance(optional_groups, dict):
            for group_name in sorted(optional_groups):
                values = optional_groups[group_name]
                if not isinstance(values, list):
                    continue
                role = _python_group_role(group_name)
                for item in values:
                    if not isinstance(item, str):
                        continue
                    raw_name = _parse_requirement_name(item)
                    if raw_name is None:
                        warnings.append(
                            _warning(source_path, "python_requirement_unparsed", item)
                        )
                        continue
                    declarations.append(
                        _declaration(
                            source_path=source_path,
                            source_kind=source_kind,
                            ecosystem="python",
                            raw_name=raw_name,
                            role=role,
                            version_spec=item,
                            group_name=group_name,
                            declaration_text=item,
                        )
                    )

    tool = data.get("tool", {})
    if isinstance(tool, dict):
        poetry = tool.get("poetry", {})
        if isinstance(poetry, dict):
            _append_mapping_declarations(
                declarations=declarations,
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="python",
                mapping=poetry.get("dependencies", {}),
                role="runtime",
            )
            _append_mapping_declarations(
                declarations=declarations,
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="python",
                mapping=poetry.get("dev-dependencies", {}),
                role="development",
                group_name="dev",
            )
            groups = poetry.get("group", {})
            if isinstance(groups, dict):
                for group_name in sorted(groups):
                    group = groups[group_name]
                    if not isinstance(group, dict):
                        continue
                    _append_mapping_declarations(
                        declarations=declarations,
                        source_path=source_path,
                        source_kind=source_kind,
                        ecosystem="python",
                        mapping=group.get("dependencies", {}),
                        role=_python_group_role(group_name),
                        group_name=group_name,
                    )

    return declarations, warnings


def _role_from_requirements_filename(path: Path) -> HistoryDependencyRole:
    name = path.name.lower()
    if "test" in name:
        return "test"
    if any(token in name for token in ("dev", "docs", "lint", "type")):
        return "development"
    return "runtime"


def _parse_requirements(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    warnings: list[HistoryDependencyWarning] = []
    role = _role_from_requirements_filename(source_path)
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(
            ("-r", "--requirement", "-c", "--constraint", "-e", "--editable")
        ):
            warnings.append(_warning(source_path, "requirements_include_skipped", line))
            continue
        raw_name = _parse_requirement_name(line)
        if raw_name is None:
            warnings.append(_warning(source_path, "python_requirement_unparsed", line))
            continue
        declarations.append(
            _declaration(
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="python",
                raw_name=raw_name,
                role=role,
                version_spec=line,
                declaration_text=line,
            )
        )
    return declarations, warnings


def _parse_pipfile(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return declarations, [_warning(source_path, "toml_parse_failed", str(exc))]

    _append_mapping_declarations(
        declarations=declarations,
        source_path=source_path,
        source_kind=source_kind,
        ecosystem="python",
        mapping=data.get("packages", {}),
        role="runtime",
    )
    _append_mapping_declarations(
        declarations=declarations,
        source_path=source_path,
        source_kind=source_kind,
        ecosystem="python",
        mapping=data.get("dev-packages", {}),
        role="development",
        group_name="dev",
    )
    return declarations, []


def _parse_pipfile_lock(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return declarations, [_warning(source_path, "json_parse_failed", str(exc))]

    if not isinstance(data, dict):
        return declarations, [
            _warning(source_path, "json_shape_invalid", "expected object")
        ]
    _append_mapping_declarations(
        declarations=declarations,
        source_path=source_path,
        source_kind=source_kind,
        ecosystem="python",
        mapping=data.get("default", {}),
        role="runtime",
    )
    _append_mapping_declarations(
        declarations=declarations,
        source_path=source_path,
        source_kind=source_kind,
        ecosystem="python",
        mapping=data.get("develop", {}),
        role="development",
        group_name="develop",
    )
    return declarations, []


def _parse_package_json(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return declarations, [_warning(source_path, "json_parse_failed", str(exc))]

    if not isinstance(data, dict):
        return declarations, [
            _warning(source_path, "json_shape_invalid", "expected object")
        ]
    for key, role in (
        ("dependencies", "runtime"),
        ("devDependencies", "development"),
        ("peerDependencies", "peer"),
        ("optionalDependencies", "optional"),
    ):
        _append_mapping_declarations(
            declarations=declarations,
            source_path=source_path,
            source_kind=source_kind,
            ecosystem="javascript",
            mapping=data.get(key, {}),
            role=cast(HistoryDependencyRole, role),
            group_name=key,
        )
    bundled = data.get("bundledDependencies") or data.get("bundleDependencies") or []
    if isinstance(bundled, list):
        for item in bundled:
            if isinstance(item, str):
                declarations.append(
                    _declaration(
                        source_path=source_path,
                        source_kind=source_kind,
                        ecosystem="javascript",
                        raw_name=item,
                        role="build",
                        group_name="bundledDependencies",
                        declaration_text=item,
                    )
                )
    return declarations, []


def _parse_package_lock(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return declarations, [_warning(source_path, "json_parse_failed", str(exc))]
    if not isinstance(data, dict):
        return declarations, [
            _warning(source_path, "json_shape_invalid", "expected object")
        ]

    root_package = (
        data.get("packages", {}).get("", {})
        if isinstance(data.get("packages"), dict)
        else {}
    )
    if isinstance(root_package, dict):
        for key, role in (
            ("dependencies", "runtime"),
            ("devDependencies", "development"),
        ):
            mapping = root_package.get(key, {})
            if isinstance(mapping, dict):
                for raw_name in sorted(mapping):
                    value = mapping[raw_name]
                    declarations.append(
                        _declaration(
                            source_path=source_path,
                            source_kind=source_kind,
                            ecosystem="javascript",
                            raw_name=raw_name,
                            role=cast(HistoryDependencyRole, role),
                            version_spec=str(value) if value is not None else None,
                            group_name=key,
                            declaration_text=(
                                str(value) if value is not None else raw_name
                            ),
                        )
                    )
    return declarations, []


def _parse_cargo_toml(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        return declarations, [_warning(source_path, "toml_parse_failed", str(exc))]

    for key, role in (
        ("dependencies", "runtime"),
        ("dev-dependencies", "test"),
        ("build-dependencies", "build"),
    ):
        _append_mapping_declarations(
            declarations=declarations,
            source_path=source_path,
            source_kind=source_kind,
            ecosystem="rust",
            mapping=data.get(key, {}),
            role=cast(HistoryDependencyRole, role),
            group_name=key,
        )
    return declarations, []


def _parse_go_mod(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    warnings: list[HistoryDependencyWarning] = []

    seen_lines: set[str] = set()
    for block_match in _GO_REQUIRE_BLOCK_RE.finditer(content):
        for raw_line in block_match.group("body").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            seen_lines.add(raw_line)
            line_match = _GO_REQUIRE_LINE_RE.match(line)
            if line_match is None:
                warnings.append(_warning(source_path, "go_require_unparsed", line))
                continue
            if "indirect" in line_match.group("rest"):
                continue
            name = line_match.group("name")
            version = line_match.group("version")
            declarations.append(
                _declaration(
                    source_path=source_path,
                    source_kind=source_kind,
                    ecosystem="go",
                    raw_name=name,
                    role="runtime",
                    version_spec=version,
                    declaration_text=line,
                )
            )
    for raw_line in content.splitlines():
        if raw_line in seen_lines:
            continue
        stripped = raw_line.strip()
        if not stripped.startswith("require "):
            continue
        remainder = stripped.removeprefix("require ")
        line_match = _GO_REQUIRE_LINE_RE.match(remainder)
        if line_match is None:
            warnings.append(_warning(source_path, "go_require_unparsed", stripped))
            continue
        if "indirect" in line_match.group("rest"):
            continue
        declarations.append(
            _declaration(
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="go",
                raw_name=line_match.group("name"),
                role="runtime",
                version_spec=line_match.group("version"),
                declaration_text=stripped,
            )
        )
    return declarations, warnings


def _xml_local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _xml_children(element: ET.Element, local_name: str) -> list[ET.Element]:
    return [
        child for child in list(element) if _xml_local_name(child.tag) == local_name
    ]


def _xml_find_text(element: ET.Element, local_name: str) -> str | None:
    for child in list(element):
        if _xml_local_name(child.tag) == local_name and child.text is not None:
            return _normalize_text(child.text)
    return None


def _parse_pom_xml(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        return declarations, [_warning(source_path, "xml_parse_failed", str(exc))]

    for child in _xml_children(root, "dependencies"):
        for dependency in _xml_children(child, "dependency"):
            group_id = _xml_find_text(dependency, "groupId")
            artifact_id = _xml_find_text(dependency, "artifactId")
            if not group_id or not artifact_id:
                continue
            version = _xml_find_text(dependency, "version")
            scope = (_xml_find_text(dependency, "scope") or "compile").lower()
            optional = (
                _xml_find_text(dependency, "optional") or "false"
            ).lower() == "true"
            role: HistoryDependencyRole
            if optional:
                role = "optional"
            elif scope == "test":
                role = "test"
            elif scope in {"provided", "system"}:
                role = "build"
            else:
                role = "runtime"
            name = f"{group_id}:{artifact_id}"
            declarations.append(
                _declaration(
                    source_path=source_path,
                    source_kind=source_kind,
                    ecosystem="jvm",
                    raw_name=name,
                    role=role,
                    version_spec=version,
                    group_name=scope,
                    declaration_text=name,
                )
            )

    build = next(
        (child for child in list(root) if _xml_local_name(child.tag) == "build"), None
    )
    if build is not None:
        for plugins in _xml_children(build, "plugins"):
            for plugin in _xml_children(plugins, "plugin"):
                group_id = _xml_find_text(plugin, "groupId") or "plugin"
                artifact_id = _xml_find_text(plugin, "artifactId")
                if artifact_id is None:
                    continue
                version = _xml_find_text(plugin, "version")
                name = f"{group_id}:{artifact_id}"
                declarations.append(
                    _declaration(
                        source_path=source_path,
                        source_kind=source_kind,
                        ecosystem="jvm",
                        raw_name=name,
                        role="plugin",
                        version_spec=version,
                        group_name="plugin",
                        declaration_text=name,
                    )
                )
    return declarations, []


def _gradle_role(config_name: str) -> HistoryDependencyRole:
    normalized = config_name.lower()
    if normalized in _GRADLE_ROLE_MAP:
        return _GRADLE_ROLE_MAP[normalized]
    if normalized.startswith("test"):
        return "test"
    if normalized.endswith("classpath") or "plugin" in normalized:
        return "plugin"
    if normalized in {"implementation", "api", "runtimeonly"}:
        return "runtime"
    return "unknown"


def _parse_gradle(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    warnings: list[HistoryDependencyWarning] = []
    for match in _GRADLE_DECLARATION_RE.finditer(content):
        config_name = match.group("config")
        coords = match.group("coords")
        if coords.startswith((":", "project(", "libs.")):
            warnings.append(_warning(source_path, "gradle_dynamic_skipped", coords))
            continue
        parts = coords.split(":")
        if len(parts) < 2:
            warnings.append(_warning(source_path, "gradle_coords_unparsed", coords))
            continue
        name = f"{parts[0]}:{parts[1]}"
        version = parts[2] if len(parts) > 2 else None
        declarations.append(
            _declaration(
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="jvm",
                raw_name=name,
                role=_gradle_role(config_name),
                version_spec=version,
                group_name=config_name,
                declaration_text=coords,
            )
        )
    for match in _GRADLE_DYNAMIC_RE.finditer(content):
        warnings.append(
            _warning(
                source_path,
                "gradle_dynamic_skipped",
                f"{match.group('config')}({match.group('expr')})",
            )
        )
    return declarations, warnings


def _parse_csproj(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        return declarations, [_warning(source_path, "xml_parse_failed", str(exc))]

    for package_ref in root.iter():
        if _xml_local_name(package_ref.tag) != "PackageReference":
            continue
        name = package_ref.attrib.get("Include") or package_ref.attrib.get("Update")
        if not name:
            continue
        version = package_ref.attrib.get("Version") or _xml_find_text(
            package_ref, "Version"
        )
        private_assets = (
            package_ref.attrib.get("PrivateAssets")
            or _xml_find_text(package_ref, "PrivateAssets")
            or ""
        ).lower()
        role: HistoryDependencyRole = "build" if "all" in private_assets else "unknown"
        declarations.append(
            _declaration(
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="dotnet",
                raw_name=name,
                role=role,
                version_spec=version,
                declaration_text=name,
            )
        )
    return declarations, []


def _parse_packages_lock(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return declarations, [_warning(source_path, "json_parse_failed", str(exc))]
    if not isinstance(data, dict):
        return declarations, [
            _warning(source_path, "json_shape_invalid", "expected object")
        ]
    dependencies = data.get("dependencies", {})
    if isinstance(dependencies, dict):
        for name in sorted(dependencies):
            details = dependencies[name]
            if not isinstance(details, dict):
                continue
            if str(details.get("type", "")).lower() != "direct":
                continue
            declarations.append(
                _declaration(
                    source_path=source_path,
                    source_kind=source_kind,
                    ecosystem="dotnet",
                    raw_name=name,
                    role="unknown",
                    version_spec=str(details.get("resolved", "")) or None,
                    declaration_text=name,
                )
            )
    return declarations, []


def _parse_vcpkg_json(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return declarations, [_warning(source_path, "json_parse_failed", str(exc))]
    dependencies = data.get("dependencies", []) if isinstance(data, dict) else []
    if isinstance(dependencies, list):
        for item in dependencies:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                name = item["name"]
            else:
                continue
            declarations.append(
                _declaration(
                    source_path=source_path,
                    source_kind=source_kind,
                    ecosystem="cpp",
                    raw_name=name,
                    role="runtime",
                    declaration_text=name,
                )
            )
    return declarations, []


def _parse_conanfile_txt(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    current_section: str | None = None
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if current_section not in _CONAN_SECTION_ROLE_MAP:
            continue
        raw_name = line.split("/", 1)[0]
        declarations.append(
            _declaration(
                source_path=source_path,
                source_kind=source_kind,
                ecosystem="cpp",
                raw_name=raw_name,
                role=_CONAN_SECTION_ROLE_MAP[current_section],
                version_spec=line,
                group_name=current_section,
                declaration_text=line,
            )
        )
    return declarations, []


def _parse_conanfile_py(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations: list[HistoryDependencyDeclaration] = []
    for match in _CONAN_ASSIGNMENT_RE.finditer(content):
        group_name = match.group("name")
        role = _CONAN_SECTION_ROLE_MAP[group_name]
        for item in _QUOTED_ITEM_RE.findall(match.group("value")):
            raw_name = item.split("/", 1)[0]
            declarations.append(
                _declaration(
                    source_path=source_path,
                    source_kind=source_kind,
                    ecosystem="cpp",
                    raw_name=raw_name,
                    role=role,
                    version_spec=item,
                    group_name=group_name,
                    declaration_text=item,
                )
            )
    return declarations, []


def _parse_meson_build(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations = [
        _declaration(
            source_path=source_path,
            source_kind=source_kind,
            ecosystem="cpp",
            raw_name=match.group("name"),
            role="runtime",
            declaration_text=match.group("name"),
        )
        for match in _MESON_DEP_RE.finditer(content)
    ]
    return declarations, []


def _parse_cmake_lists(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations = [
        _declaration(
            source_path=source_path,
            source_kind=source_kind,
            ecosystem="cpp",
            raw_name=match.group("name"),
            role="build",
            declaration_text=match.group("name"),
        )
        for match in _CMAKE_PACKAGE_RE.finditer(content)
    ]
    return declarations, []


def _parse_workspace(
    source_path: Path,
    content: str,
    source_kind: HistoryDependencySourceKind,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    declarations = [
        _declaration(
            source_path=source_path,
            source_kind=source_kind,
            ecosystem="generic",
            raw_name=match.group("name"),
            role="build",
            declaration_text=match.group("name"),
        )
        for match in _WORKSPACE_REPO_RE.finditer(content)
    ]
    return declarations, []


def _parse_source(
    *,
    source_path: Path,
    ecosystem: str,
    content: str,
) -> tuple[list[HistoryDependencyDeclaration], list[HistoryDependencyWarning]]:
    source_kind = _source_kind(source_path)
    name = source_path.name
    if name == "pyproject.toml":
        return _parse_pyproject(source_path, content, source_kind)
    if name.startswith("requirements") and name.endswith(".txt"):
        return _parse_requirements(source_path, content, source_kind)
    if name == "Pipfile":
        return _parse_pipfile(source_path, content, source_kind)
    if name == "Pipfile.lock":
        return _parse_pipfile_lock(source_path, content, source_kind)
    if name == "package.json":
        return _parse_package_json(source_path, content, source_kind)
    if name == "package-lock.json":
        return _parse_package_lock(source_path, content, source_kind)
    if name == "Cargo.toml":
        return _parse_cargo_toml(source_path, content, source_kind)
    if name == "go.mod":
        return _parse_go_mod(source_path, content, source_kind)
    if name == "pom.xml":
        return _parse_pom_xml(source_path, content, source_kind)
    if name in {"build.gradle", "build.gradle.kts"}:
        return _parse_gradle(source_path, content, source_kind)
    if source_path.suffix == ".csproj":
        return _parse_csproj(source_path, content, source_kind)
    if name == "packages.lock.json":
        return _parse_packages_lock(source_path, content, source_kind)
    if name == "vcpkg.json":
        return _parse_vcpkg_json(source_path, content, source_kind)
    if name == "conanfile.txt":
        return _parse_conanfile_txt(source_path, content, source_kind)
    if name == "conanfile.py":
        return _parse_conanfile_py(source_path, content, source_kind)
    if name == "meson.build":
        return _parse_meson_build(source_path, content, source_kind)
    if name == "CMakeLists.txt":
        return _parse_cmake_lists(source_path, content, source_kind)
    if name == "WORKSPACE":
        return _parse_workspace(source_path, content, source_kind)
    return [], [
        _warning(
            source_path,
            "metadata_only_skipped",
            f"H7 does not parse direct dependencies from {name} yet",
        )
    ]


def _entry_sort_key(entry: HistoryDependencyEntry) -> tuple[str, str]:
    return (entry.ecosystem, entry.normalized_name)


def _warning_sort_key(warning: HistoryDependencyWarning) -> tuple[str, str, str]:
    return (warning.source_path.as_posix(), warning.code, warning.message)


def _module_token_candidates(import_text: str) -> set[str]:
    stripped = import_text.strip().strip("'\"")
    tokens = set(_IMPORT_TOKEN_RE.findall(stripped))
    for token in list(tokens):
        if (
            token.startswith("from")
            or token.startswith("import")
            or token.startswith("use")
        ):
            pieces = token.split()
            if pieces:
                tokens.add(pieces[-1])
        if "::" in token:
            tokens.add(token.split("::", 1)[0])
        if "/" in token and not token.startswith("@"):
            tokens.add(token.split("/", 1)[0])
        if "." in token:
            tokens.add(token.split(".", 1)[0])
    return {
        token.strip()
        for token in tokens
        if token.strip() and token not in {"from", "import", "use", "as"}
    }


def _normalized_import_candidates(ecosystem: str, import_text: str) -> set[str]:
    candidates: set[str] = set()
    for token in _module_token_candidates(import_text):
        normalized = _normalize_name(ecosystem, token)
        candidates.add(normalized)
        if ecosystem in {"python", "rust"}:
            candidates.add(normalized.replace("-", "_"))
        if token.startswith("@"):
            candidates.add(token.lower())
    return candidates


def _entry_match_tokens(entry: HistoryDependencyEntry) -> set[str]:
    tokens = {
        entry.normalized_name,
        _normalize_name(entry.ecosystem, entry.display_name),
    }
    if entry.ecosystem in {"python", "rust"}:
        tokens.add(entry.normalized_name.replace("-", "_"))
    return {token for token in tokens if token}


def _usage_matches(module: HistoryModuleConcept, entry: HistoryDependencyEntry) -> bool:
    if entry.ecosystem not in _RELIABLE_USAGE_ECOSYSTEMS:
        return False
    module_candidates = set()
    for import_text in module.imports:
        module_candidates.update(
            _normalized_import_candidates(entry.ecosystem, import_text)
        )
    entry_tokens = _entry_match_tokens(entry)
    if entry.ecosystem == "go":
        return any(
            candidate == token
            or candidate.startswith(f"{token}/")
            or token.startswith(f"{candidate}/")
            for candidate in module_candidates
            for token in entry_tokens
        )
    return bool(module_candidates & entry_tokens)


def _related_usage(
    entry: HistoryDependencyEntry,
    active_modules: list[HistoryModuleConcept],
) -> tuple[list[str], list[str], list[str]]:
    related_module_ids: set[str] = set()
    related_subsystem_ids: set[str] = set(entry.related_subsystem_ids)
    usage_signals: set[str] = set()
    for module in active_modules:
        if not _usage_matches(module, entry):
            continue
        related_module_ids.add(module.concept_id)
        if module.subsystem_id is not None:
            related_subsystem_ids.add(module.subsystem_id)
        usage_signals.add(f"{module.path.as_posix()} imports {entry.display_name}")
    return (
        sorted(related_module_ids),
        sorted(related_subsystem_ids),
        sorted(usage_signals),
    )


def _section_target(
    roles: set[HistoryDependencyRole],
) -> HistoryDependencySectionTarget:
    if {"runtime", "peer", "optional"} & roles:
        return "dependencies"
    return "build_development_infrastructure"


def _summarize_entry(
    *,
    entry: HistoryDependencyEntry,
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
) -> tuple[HistoryDependencyEntry, HistoryDependencyWarning | None]:
    if entry.declarations and all(
        item.source_kind == "lockfile" for item in entry.declarations
    ):
        warning = _warning(
            entry.source_manifest_paths[0],
            "lockfile_only_summary_skipped",
            f"{entry.display_name} is documented from lockfile-only evidence; summary left as TBD",
        )
        return (
            entry.model_copy(
                update={
                    "general_description": "TBD",
                    "project_usage_description": "TBD",
                    "summary_status": "tbd",
                    "uncertainty": sorted({*entry.uncertainty, warning.message}),
                    "confidence": 0.0,
                }
            ),
            warning,
        )

    system_prompt, user_prompt = build_dependency_summary_prompt(entry)
    try:
        response = llm_client.generate_structured(
            StructuredGenerationRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=HistoryDependencySummary,
                model_name=model_name,
                temperature=temperature,
            )
        )
        summary = HistoryDependencySummary.model_validate(
            response.content.model_dump(mode="json")
        )
    except EngLLMError as exc:
        warning = _warning(
            entry.source_manifest_paths[0],
            "llm_summary_failed",
            f"Dependency summary generation failed for {entry.display_name}: {exc}",
        )
        return (
            entry.model_copy(
                update={
                    "general_description": "TBD",
                    "project_usage_description": "TBD",
                    "summary_status": "llm_failed",
                    "uncertainty": sorted({*entry.uncertainty, warning.message}),
                    "confidence": 0.0,
                }
            ),
            warning,
        )

    return (
        entry.model_copy(
            update={
                "general_description": summary.general_description,
                "project_usage_description": summary.project_usage_description,
                "summary_status": "documented",
                "uncertainty": sorted({*entry.uncertainty, *summary.uncertainty}),
                "confidence": summary.confidence,
            }
        ),
        None,
    )


def _active_dependency_concepts(
    checkpoint_model: HistoryCheckpointModel,
) -> list[HistoryDependencyConcept]:
    return [
        concept
        for concept in checkpoint_model.dependencies
        if concept.lifecycle_status == "active"
    ]


def build_dependency_inventory(
    *,
    repo_root: Path,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    checkpoint_model: HistoryCheckpointModel,
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
    read_file_at_commit: Callable[[Path, str, Path], str],
) -> HistoryDependencyInventory:
    """Build the H7 dependency inventory for one checkpoint."""

    active_modules = [
        module
        for module in checkpoint_model.modules
        if module.lifecycle_status == "active"
    ]
    warnings: list[HistoryDependencyWarning] = []
    entries_by_id: dict[str, HistoryDependencyEntry] = {}

    for concept in _active_dependency_concepts(checkpoint_model):
        try:
            content = read_file_at_commit(repo_root, target_commit, concept.path)
        except GitError as exc:
            warnings.append(
                _warning(
                    concept.path,
                    "read_at_commit_failed",
                    f"Failed reading {concept.path.as_posix()} at {target_commit}: {exc}",
                )
            )
            continue
        parsed_declarations, parsed_warnings = _parse_source(
            source_path=concept.path,
            ecosystem=concept.ecosystem,
            content=content,
        )
        warnings.extend(parsed_warnings)
        for declaration in parsed_declarations:
            dependency_id = _dependency_id(
                concept.ecosystem,
                declaration.normalized_name,
            )
            if dependency_id not in entries_by_id:
                entries_by_id[dependency_id] = HistoryDependencyEntry(
                    dependency_id=dependency_id,
                    display_name=declaration.raw_name,
                    normalized_name=declaration.normalized_name,
                    ecosystem=concept.ecosystem,
                    section_target="build_development_infrastructure",
                )
            entry = entries_by_id[dependency_id]
            if declaration not in entry.declarations:
                entry.declarations.append(declaration)
            if concept.path not in entry.source_manifest_paths:
                entry.source_manifest_paths.append(concept.path)
            concept_id = concept.concept_id
            if concept_id not in entry.source_dependency_concept_ids:
                entry.source_dependency_concept_ids.append(concept_id)
            entry.related_subsystem_ids = sorted(
                {*entry.related_subsystem_ids, *concept.related_subsystem_ids}
            )

    enriched_entries: list[HistoryDependencyEntry] = []
    for dependency_id in sorted(entries_by_id):
        entry = entries_by_id[dependency_id]
        entry.declarations = sorted(
            entry.declarations,
            key=lambda item: (
                item.source_path.as_posix(),
                item.source_kind,
                item.role,
                item.raw_name.lower(),
                item.version_spec or "",
            ),
        )
        entry.source_manifest_paths = sorted(
            entry.source_manifest_paths,
            key=lambda item: item.as_posix(),
        )
        entry.source_dependency_concept_ids = sorted(
            entry.source_dependency_concept_ids
        )
        entry.scope_roles = sorted({item.role for item in entry.declarations})
        entry.section_target = _section_target(set(entry.scope_roles))
        related_module_ids, related_subsystem_ids, usage_signals = _related_usage(
            entry,
            active_modules,
        )
        entry.related_module_ids = related_module_ids
        entry.related_subsystem_ids = related_subsystem_ids
        entry.usage_signals = usage_signals
        summarized_entry, summary_warning = _summarize_entry(
            entry=entry,
            llm_client=llm_client,
            model_name=model_name,
            temperature=temperature,
        )
        if summary_warning is not None:
            warnings.append(summary_warning)
        enriched_entries.append(summarized_entry)

    return HistoryDependencyInventory(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        entries=sorted(enriched_entries, key=_entry_sort_key),
        warnings=sorted(warnings, key=_warning_sort_key),
    )


def link_dependency_inventory_to_checkpoint_model(
    checkpoint_model: HistoryCheckpointModel,
    dependency_inventory: HistoryDependencyInventory,
) -> HistoryCheckpointModel:
    """Attach dependency entry links back onto dependency concepts."""

    dependency_ids_by_concept: dict[str, list[str]] = defaultdict(list)
    for entry in dependency_inventory.entries:
        for concept_id in entry.source_dependency_concept_ids:
            dependency_ids_by_concept[concept_id].append(entry.dependency_id)

    dependencies = [
        concept.model_copy(
            update={
                "documented_dependency_ids": sorted(
                    set(dependency_ids_by_concept.get(concept.concept_id, []))
                ),
                "documented_dependency_count": len(
                    set(dependency_ids_by_concept.get(concept.concept_id, []))
                ),
            }
        )
        for concept in checkpoint_model.dependencies
    ]
    return checkpoint_model.model_copy(update={"dependencies": dependencies})


__all__ = [
    "build_dependency_inventory",
    "dependency_inventory_path",
    "link_dependency_inventory_to_checkpoint_model",
]
