"""Language-aware tree-sitter analyzers for repository data extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import CodeUnitSummary, InterfaceSummary, SourceLanguage

LANGUAGE_BY_SUFFIX: dict[str, SourceLanguage] = {
    ".py": "python",
    ".java": "java",
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
}

_IDENTIFIER_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "type_identifier",
    "qualified_identifier",
    "namespace_identifier",
    "property_identifier",
}


class LanguageAnalyzer(Protocol):
    """Analyzer contract for one programming language."""

    language: SourceLanguage
    supported_suffixes: tuple[str, ...]

    def supports(self, path: Path) -> bool:
        """Return whether the analyzer supports this file extension."""

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        """Build deterministic code and interface summaries."""

    def signature_changes(self, changed_lines: list[str]) -> list[str]:
        """Return signature-like changed lines."""

    def dependency_changes(self, changed_lines: list[str]) -> list[str]:
        """Return dependency-like changed lines."""

    def is_comment_line(self, line: str) -> bool:
        """Return whether a changed line is comment-only."""


class TSNode(Protocol):
    """Minimal tree-sitter node protocol used by analyzers."""

    start_byte: int
    end_byte: int
    type: str
    children: list[TSNode]

    def child_by_field_name(self, field_name: str) -> TSNode | None:
        """Return child for a named grammar field."""


class TSTree(Protocol):
    """Minimal tree-sitter tree protocol used by analyzers."""

    root_node: TSNode


class TSParser(Protocol):
    """Minimal tree-sitter parser protocol used by analyzers."""

    def parse(self, source_bytes: bytes) -> TSTree:
        """Parse bytes into a syntax tree."""


_PARSER_CACHE: dict[str, TSParser] = {}
_ANALYZER_CACHE: list[LanguageAnalyzer] | None = None


def _get_parser(parser_name: str) -> TSParser:
    parser = _PARSER_CACHE.get(parser_name)
    if parser is not None:
        return parser

    try:
        from tree_sitter_languages import get_parser
    except Exception as exc:  # pragma: no cover - import-time environment
        raise AnalysisError(
            "tree-sitter support requires 'tree-sitter-languages'. Install project dependencies first."
        ) from exc

    try:
        parser = get_parser(parser_name)
    except Exception as exc:
        raise AnalysisError(
            f"Failed to initialize tree-sitter parser for '{parser_name}': {exc}"
        ) from exc

    typed_parser = cast(TSParser, parser)
    _PARSER_CACHE[parser_name] = typed_parser
    return typed_parser


def _walk(node: TSNode) -> Iterable[TSNode]:
    stack: list[TSNode] = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in reversed(current.children):
            stack.append(child)


def _node_text(source_bytes: bytes, node: TSNode) -> str:
    start = int(node.start_byte)
    end = int(node.end_byte)
    return source_bytes[start:end].decode("utf-8", errors="ignore")


def _field_text(source_bytes: bytes, node: TSNode, field_name: str) -> str | None:
    child = node.child_by_field_name(field_name)
    if child is None:
        return None
    text = _node_text(source_bytes, child).strip()
    return text or None


def _first_identifier(source_bytes: bytes, node: TSNode) -> str | None:
    for item in _walk(node):
        if item.type in _IDENTIFIER_NODE_TYPES:
            value = _node_text(source_bytes, item).strip()
            if value:
                return value
    return None


def _append_module_interface(
    interfaces: list[InterfaceSummary],
    path: Path,
    language: SourceLanguage,
    classes: set[str],
    functions: set[str],
) -> None:
    members = sorted({*classes, *functions})
    if not members:
        return

    interfaces.append(
        InterfaceSummary(
            name=path.stem,
            kind="module",
            language=language,
            source_path=path,
            members=members,
        )
    )


def detect_language(path: Path) -> SourceLanguage:
    """Map file extension to normalized language name."""

    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "unknown")


class _BaseAnalyzer:
    language: SourceLanguage = "unknown"
    parser_name = ""
    supported_suffixes: tuple[str, ...] = ()

    signature_patterns: tuple[re.Pattern[str], ...] = ()
    dependency_patterns: tuple[re.Pattern[str], ...] = ()

    comment_prefixes: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._parser = _get_parser(self.parser_name)

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_suffixes

    def _parse(self, source_text: str) -> tuple[TSTree, bytes]:
        source_bytes = source_text.encode("utf-8", errors="ignore")
        tree = self._parser.parse(source_bytes)
        return tree, source_bytes

    def signature_changes(self, changed_lines: list[str]) -> list[str]:
        matched: list[str] = []
        for line in changed_lines:
            stripped = line.strip()
            if any(pattern.match(stripped) for pattern in self.signature_patterns):
                matched.append(stripped)
        return matched

    def dependency_changes(self, changed_lines: list[str]) -> list[str]:
        matched: list[str] = []
        for line in changed_lines:
            stripped = line.strip()
            if any(pattern.match(stripped) for pattern in self.dependency_patterns):
                matched.append(stripped)
        return matched

    def is_comment_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        return any(stripped.startswith(prefix) for prefix in self.comment_prefixes)


class PythonAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for Python files."""

    language: SourceLanguage = "python"
    parser_name = "python"
    supported_suffixes: tuple[str, ...] = (".py",)

    signature_patterns = (re.compile(r"^(def|class)\s+\w+"),)
    dependency_patterns = (re.compile(r"^(from\s+\S+\s+import\s+.+|import\s+.+)$"),)

    comment_prefixes = ("#",)

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        class_nodes: list[TSNode] = []

        for node in _walk(tree.root_node):
            if node.type == "function_definition":
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type == "class_definition":
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
                    class_nodes.append(node)
            elif node.type in {"import_statement", "import_from_statement"}:
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        public_functions = {name for name in functions if not name.startswith("_")}

        for class_node in class_nodes:
            class_name = _field_text(source_bytes, class_node, "name")
            if not class_name or class_name.startswith("_"):
                continue
            members = sorted(
                {
                    _field_text(source_bytes, child, "name") or ""
                    for child in _walk(class_node)
                    if child.type == "function_definition"
                }
            )
            members = [name for name in members if name and not name.startswith("_")]
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=members,
                )
            )

        for function_name in sorted(public_functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        public_classes = {item.name for item in interfaces if item.kind == "class"}
        _append_module_interface(
            interfaces,
            path,
            self.language,
            public_classes,
            public_functions,
        )

        return summary, interfaces


class JavaAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for Java files."""

    language: SourceLanguage = "java"
    parser_name = "java"
    supported_suffixes: tuple[str, ...] = (".java",)

    signature_patterns = (
        re.compile(r"^(class|interface|enum)\s+\w+"),
        re.compile(
            r"^(public|protected|private|static|final|abstract|synchronized|native|strictfp|\s)*"
            r"[A-Za-z_][\w<>,\[\]\s]*\s+[A-Za-z_]\w*\s*\([^;]*\)\s*(\{|throws|$)"
        ),
    )
    dependency_patterns = (re.compile(r"^(import|package)\s+[A-Za-z0-9_.*]+\s*;"),)

    comment_prefixes = ("//", "/*", "*", "*/")

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        type_nodes: list[TSNode] = []

        for node in _walk(tree.root_node):
            if node.type in {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            }:
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
                    type_nodes.append(node)
            elif node.type in {"method_declaration", "constructor_declaration"}:
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type in {"import_declaration", "package_declaration"}:
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for type_node in type_nodes:
            type_name = _field_text(source_bytes, type_node, "name")
            if not type_name:
                continue
            members = sorted(
                {
                    _field_text(source_bytes, child, "name") or ""
                    for child in _walk(type_node)
                    if child.type in {"method_declaration", "constructor_declaration"}
                }
            )
            interfaces.append(
                InterfaceSummary(
                    name=type_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[item for item in members if item],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class CppAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for C/C++ files."""

    language: SourceLanguage = "cpp"
    parser_name = "cpp"
    supported_suffixes: tuple[str, ...] = (".c", ".cc", ".cpp", ".h", ".hpp")

    signature_patterns = (
        re.compile(r"^(class|struct)\s+\w+"),
        re.compile(
            r"^(template\s*<.*>\s*)?[A-Za-z_:~][\w:\<\>\*&\s]*\s+[A-Za-z_~]\w*\s*\([^;{}]*\)\s*(const)?\s*(\{|;)$"
        ),
    )
    dependency_patterns = (
        re.compile(r"^#\s*(include|import)\s*[<\"].+[>\"]"),
        re.compile(r"^using\s+namespace\s+[A-Za-z_][\w:]*\s*;"),
    )

    comment_prefixes = ("//", "/*", "*", "*/")

    def _extract_cpp_callable_name(self, declarator_text: str) -> str | None:
        matches = re.findall(r"([~A-Za-z_][\w:]*)\s*\(", declarator_text)
        if not matches:
            return None
        candidate = cast(str, matches[-1])
        return candidate.split("::")[-1]

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        for node in _walk(tree.root_node):
            if node.type in {"class_specifier", "struct_specifier"}:
                name = _field_text(source_bytes, node, "name") or _first_identifier(
                    source_bytes, node
                )
                if name:
                    classes.add(name)
            elif node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                declarator_text = (
                    _node_text(source_bytes, declarator)
                    if declarator is not None
                    else _node_text(source_bytes, node)
                )
                name = self._extract_cpp_callable_name(declarator_text)
                if name:
                    functions.add(name)
            elif node.type == "preproc_include":
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for class_name in sorted(classes):
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class JavaScriptAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for JavaScript files."""

    language: SourceLanguage = "javascript"
    parser_name = "javascript"
    supported_suffixes: tuple[str, ...] = (".js", ".mjs", ".cjs")

    signature_patterns = (
        re.compile(r"^(export\s+)?(async\s+)?function\s+\w+"),
        re.compile(r"^class\s+\w+"),
        re.compile(r"^(const|let|var)\s+\w+\s*=\s*(async\s*)?\([^)]*\)\s*=>"),
    )
    dependency_patterns = (
        re.compile(r"^import\s+.+\s+from\s+['\"].+['\"]"),
        re.compile(r"^const\s+\w+\s*=\s*require\(['\"].+['\"]\)"),
    )

    comment_prefixes = ("//", "/*", "*", "*/")

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        for node in _walk(tree.root_node):
            if node.type in {"function_declaration", "generator_function_declaration"}:
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type == "class_declaration":
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
            elif node.type == "method_definition":
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type == "import_statement":
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for class_name in sorted(classes):
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class TypeScriptAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for TypeScript files."""

    language: SourceLanguage = "typescript"
    parser_name = "typescript"
    supported_suffixes: tuple[str, ...] = (".ts", ".tsx")

    signature_patterns = (
        re.compile(r"^(export\s+)?(async\s+)?function\s+\w+"),
        re.compile(r"^(export\s+)?(class|interface|type)\s+\w+"),
        re.compile(r"^(const|let|var)\s+\w+\s*:\s*.+="),
    )
    dependency_patterns = (
        re.compile(r"^import\s+.+\s+from\s+['\"].+['\"]"),
        re.compile(r"^export\s+.+\s+from\s+['\"].+['\"]"),
    )

    comment_prefixes = ("//", "/*", "*", "*/")

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        for node in _walk(tree.root_node):
            if node.type in {
                "function_declaration",
                "method_definition",
                "method_signature",
            }:
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type in {
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            }:
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
            elif node.type in {"import_statement", "export_statement"}:
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for class_name in sorted(classes):
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class GoAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for Go files."""

    language: SourceLanguage = "go"
    parser_name = "go"
    supported_suffixes: tuple[str, ...] = (".go",)

    signature_patterns = (
        re.compile(r"^func\s+\w+\s*\("),
        re.compile(r"^func\s*\([^)]*\)\s*\w+\s*\("),
        re.compile(r"^type\s+\w+\s+(struct|interface)\b"),
    )
    dependency_patterns = (
        re.compile(r"^import\s+\("),
        re.compile(r"^import\s+['\"].+['\"]"),
    )

    comment_prefixes = ("//", "/*", "*", "*/")

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        for node in _walk(tree.root_node):
            if node.type in {"function_declaration", "method_declaration"}:
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type == "type_spec":
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
            elif node.type in {"import_declaration", "import_spec"}:
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for class_name in sorted(classes):
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class RustAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for Rust files."""

    language: SourceLanguage = "rust"
    parser_name = "rust"
    supported_suffixes: tuple[str, ...] = (".rs",)

    signature_patterns = (
        re.compile(r"^(pub\s+)?fn\s+\w+"),
        re.compile(r"^(pub\s+)?(struct|enum|trait)\s+\w+"),
        re.compile(r"^impl\s+"),
    )
    dependency_patterns = (
        re.compile(r"^use\s+[A-Za-z0-9_:{}*,\s]+;"),
        re.compile(r"^extern\s+crate\s+\w+;"),
    )

    comment_prefixes = ("//", "/*", "*", "*/")

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        for node in _walk(tree.root_node):
            if node.type == "function_item":
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type in {"struct_item", "enum_item", "trait_item", "impl_item"}:
                name = _field_text(source_bytes, node, "name") or _first_identifier(
                    source_bytes, node
                )
                if name:
                    classes.add(name)
            elif node.type in {"use_declaration", "extern_crate_declaration"}:
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for class_name in sorted(classes):
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class CSharpAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for C# files."""

    language: SourceLanguage = "csharp"
    parser_name = "c_sharp"
    supported_suffixes: tuple[str, ...] = (".cs",)

    signature_patterns = (
        re.compile(r"^(class|interface|struct|enum|record)\s+\w+"),
        re.compile(
            r"^(public|private|protected|internal|static|virtual|override|sealed|abstract|partial|\s)+"
            r"[A-Za-z_][\w<>,\[\]\s]*\s+[A-Za-z_]\w*\s*\([^;]*\)\s*(\{|$)"
        ),
    )
    dependency_patterns = (
        re.compile(r"^(global\s+)?using\s+[A-Za-z0-9_.]+\s*;"),
        re.compile(r"^extern\s+alias\s+\w+\s*;"),
    )

    comment_prefixes = ("//", "/*", "*", "*/")

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()

        for node in _walk(tree.root_node):
            if node.type in {
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
            }:
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
            elif node.type in {
                "method_declaration",
                "constructor_declaration",
                "local_function_statement",
            }:
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
            elif node.type in {"using_directive", "extern_alias_directive"}:
                imports.add(_node_text(source_bytes, node).strip())

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        interfaces: list[InterfaceSummary] = []
        for class_name in sorted(classes):
            interfaces.append(
                InterfaceSummary(
                    name=class_name,
                    kind="class",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        for function_name in sorted(functions):
            interfaces.append(
                InterfaceSummary(
                    name=function_name,
                    kind="function",
                    language=self.language,
                    source_path=path,
                    members=[],
                )
            )

        _append_module_interface(interfaces, path, self.language, classes, functions)
        return summary, interfaces


class UnknownAnalyzer:
    """Fallback analyzer for unsupported languages."""

    language: SourceLanguage = "unknown"
    supported_suffixes: tuple[str, ...] = ()

    _signature_patterns = (
        re.compile(r"^(def|class|interface|enum|struct|template|fn|func|impl)\b"),
    )
    _dependency_patterns = (
        re.compile(r"^(import|from\s+\S+\s+import|#\s*include|package|using|use)\b"),
    )

    def supports(self, path: Path) -> bool:
        return False

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[InterfaceSummary]]:
        raise AnalysisError(f"No analyzer registered for file '{path}'")

    def signature_changes(self, changed_lines: list[str]) -> list[str]:
        return [
            line.strip()
            for line in changed_lines
            if any(pattern.match(line.strip()) for pattern in self._signature_patterns)
        ]

    def dependency_changes(self, changed_lines: list[str]) -> list[str]:
        return [
            line.strip()
            for line in changed_lines
            if any(pattern.match(line.strip()) for pattern in self._dependency_patterns)
        ]

    def is_comment_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        return stripped.startswith(("#", "//", "/*", "*", "*/"))


def get_registered_analyzers() -> list[LanguageAnalyzer]:
    """Return concrete language analyzers supported by this project."""

    global _ANALYZER_CACHE
    if _ANALYZER_CACHE is None:
        _ANALYZER_CACHE = [
            PythonAnalyzer(),
            JavaAnalyzer(),
            CppAnalyzer(),
            JavaScriptAnalyzer(),
            TypeScriptAnalyzer(),
            GoAnalyzer(),
            RustAnalyzer(),
            CSharpAnalyzer(),
        ]
    return _ANALYZER_CACHE


def get_analyzer_for_path(path: Path) -> LanguageAnalyzer:
    """Return analyzer for a given source file path."""

    suffix = path.suffix.lower()
    for analyzer in get_registered_analyzers():
        if suffix in analyzer.supported_suffixes:
            return analyzer
    return UnknownAnalyzer()


def get_analyzer_for_language(language: SourceLanguage) -> LanguageAnalyzer:
    """Return analyzer for normalized language name."""

    for analyzer in get_registered_analyzers():
        if analyzer.language == language:
            return analyzer
    return UnknownAnalyzer()
