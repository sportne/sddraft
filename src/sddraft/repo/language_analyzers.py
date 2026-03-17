"""Language-aware tree-sitter analyzers for repository data extraction."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol, cast

from sddraft.domain.errors import AnalysisError
from sddraft.domain.models import (
    CodeUnitSummary,
    SourceLanguage,
    SymbolSummary,
)

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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        """Build deterministic code and symbol summaries."""

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
    start_point: tuple[int, int] | list[int]
    end_point: tuple[int, int] | list[int]
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
        from tree_sitter_language_pack import get_parser as get_language_pack_parser
    except Exception as exc:  # pragma: no cover - import-time environment
        raise AnalysisError(
            "tree-sitter support requires 'tree-sitter-language-pack'. Install project dependencies first."
        ) from exc

    try:
        get_parser = cast(Callable[[str], TSParser], get_language_pack_parser)
        parser = get_parser(parser_name)
    except Exception as exc:
        raise AnalysisError(
            f"Failed to initialize tree-sitter parser for '{parser_name}': {exc}"
        ) from exc

    _PARSER_CACHE[parser_name] = parser
    return parser


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


def _line_span(node: TSNode) -> tuple[int | None, int | None]:
    start_point = getattr(node, "start_point", None)
    end_point = getattr(node, "end_point", None)

    if (
        isinstance(start_point, (tuple, list))
        and len(start_point) >= 1
        and isinstance(start_point[0], int)
    ):
        start_line = int(start_point[0]) + 1
    else:
        start_line = None

    if (
        isinstance(end_point, (tuple, list))
        and len(end_point) >= 1
        and isinstance(end_point[0], int)
    ):
        end_line = int(end_point[0]) + 1
    else:
        end_line = None

    return start_line, end_line


def _qualified(owner_qualified_name: str | None, name: str) -> str:
    if owner_qualified_name:
        return f"{owner_qualified_name}.{name}"
    return name


def _append_symbol(
    *,
    symbols: list[SymbolSummary],
    seen: set[tuple[str, str, str]],
    name: str | None,
    kind: str,
    language: SourceLanguage,
    source_path: Path,
    owner_qualified_name: str | None = None,
    node: TSNode | None = None,
) -> None:
    cleaned_name = (name or "").strip()
    if not cleaned_name:
        return
    qualified_name = _qualified(owner_qualified_name, cleaned_name)
    dedupe_key = (kind, qualified_name, source_path.as_posix())
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)
    line_start, line_end = _line_span(node) if node is not None else (None, None)
    symbols.append(
        SymbolSummary(
            name=cleaned_name,
            qualified_name=qualified_name,
            kind=kind,
            language=language,
            source_path=source_path,
            owner_qualified_name=owner_qualified_name,
            line_start=line_start,
            line_end=line_end,
        )
    )


def _append_module_symbol(
    *,
    symbols: list[SymbolSummary],
    seen: set[tuple[str, str, str]],
    path: Path,
    language: SourceLanguage,
    classes: set[str],
    functions: set[str],
) -> None:
    if not classes and not functions:
        return
    _append_symbol(
        symbols=symbols,
        seen=seen,
        name=path.stem,
        kind="module",
        language=language,
        source_path=path,
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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        def walk_python(
            node: TSNode,
            owner_qualified_name: str | None = None,
            in_class: bool = False,
        ) -> None:
            if node.type in {"import_statement", "import_from_statement"}:
                imports.add(_node_text(source_bytes, node).strip())

            if node.type == "class_definition":
                class_name = _field_text(source_bytes, node, "name")
                class_owner: str | None
                if class_name:
                    classes.add(class_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=class_name,
                        kind="class",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    class_owner = _qualified(owner_qualified_name, class_name)
                else:
                    class_owner = owner_qualified_name

                for child in node.children:
                    walk_python(child, class_owner, in_class=True)
                return

            if node.type == "function_definition":
                function_name = _field_text(source_bytes, node, "name")
                function_owner: str | None
                if function_name:
                    functions.add(function_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=function_name,
                        kind="method" if in_class else "function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    function_owner = _qualified(owner_qualified_name, function_name)
                else:
                    function_owner = owner_qualified_name

                for child in node.children:
                    walk_python(child, function_owner, in_class=in_class)
                return

            for child in node.children:
                walk_python(child, owner_qualified_name, in_class=in_class)

        walk_python(tree.root_node)
        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )

        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        def walk_java(node: TSNode, owner_qualified_name: str | None = None) -> None:
            if node.type in {"import_declaration", "package_declaration"}:
                imports.add(_node_text(source_bytes, node).strip())

            if node.type in {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            }:
                type_name = _field_text(source_bytes, node, "name")
                type_owner: str | None
                if type_name:
                    classes.add(type_name)
                    kind = {
                        "class_declaration": "class",
                        "interface_declaration": "interface",
                        "enum_declaration": "enum",
                    }.get(node.type, "class")
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=type_name,
                        kind=kind,
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    type_owner = _qualified(owner_qualified_name, type_name)
                else:
                    type_owner = owner_qualified_name

                for child in node.children:
                    walk_java(child, type_owner)
                return

            if node.type in {"method_declaration", "constructor_declaration"}:
                method_name = _field_text(source_bytes, node, "name")
                method_owner: str | None
                if method_name:
                    functions.add(method_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=method_name,
                        kind="method" if owner_qualified_name else "function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    method_owner = _qualified(owner_qualified_name, method_name)
                else:
                    method_owner = owner_qualified_name

                for child in node.children:
                    walk_java(child, method_owner)
                return

            for child in node.children:
                walk_java(child, owner_qualified_name)

        walk_java(tree.root_node)
        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        for node in _walk(tree.root_node):
            if node.type in {"class_specifier", "struct_specifier"}:
                name = _field_text(source_bytes, node, "name") or _first_identifier(
                    source_bytes, node
                )
                if name:
                    classes.add(name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=name,
                        kind="struct" if node.type == "struct_specifier" else "class",
                        language=self.language,
                        source_path=path,
                        node=node,
                    )
            elif node.type == "function_definition":
                declarator = node.child_by_field_name("declarator")
                declarator_text = (
                    _node_text(source_bytes, declarator)
                    if declarator is not None
                    else _node_text(source_bytes, node)
                )
                qualified_matches = re.findall(
                    r"([~A-Za-z_][\w:]*)\s*\(",
                    declarator_text,
                )
                qualified_name = (
                    cast(str, qualified_matches[-1]) if qualified_matches else None
                )
                name = self._extract_cpp_callable_name(declarator_text)
                if name:
                    functions.add(name)
                    owner = None
                    if qualified_name and "::" in qualified_name:
                        owner = qualified_name.rsplit("::", maxsplit=1)[0]
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=name,
                        kind="method" if owner else "function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner,
                        node=node,
                    )
            elif node.type == "preproc_include":
                imports.add(_node_text(source_bytes, node).strip())

        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        def walk_javascript(
            node: TSNode, owner_qualified_name: str | None = None
        ) -> None:
            if node.type == "import_statement":
                imports.add(_node_text(source_bytes, node).strip())

            if node.type == "class_declaration":
                class_name = _field_text(source_bytes, node, "name")
                class_owner: str | None
                if class_name:
                    classes.add(class_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=class_name,
                        kind="class",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    class_owner = _qualified(owner_qualified_name, class_name)
                else:
                    class_owner = owner_qualified_name
                for child in node.children:
                    walk_javascript(child, class_owner)
                return

            if node.type in {"function_declaration", "generator_function_declaration"}:
                function_name = _field_text(source_bytes, node, "name")
                function_owner: str | None
                if function_name:
                    functions.add(function_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=function_name,
                        kind="function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    function_owner = _qualified(owner_qualified_name, function_name)
                else:
                    function_owner = owner_qualified_name
                for child in node.children:
                    walk_javascript(child, function_owner)
                return

            if node.type == "method_definition":
                method_name = _field_text(source_bytes, node, "name")
                if method_name:
                    functions.add(method_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=method_name,
                        kind="method",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                for child in node.children:
                    walk_javascript(child, owner_qualified_name)
                return

            for child in node.children:
                walk_javascript(child, owner_qualified_name)

        walk_javascript(tree.root_node)
        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        def walk_typescript(
            node: TSNode, owner_qualified_name: str | None = None
        ) -> None:
            if node.type in {"import_statement", "export_statement"}:
                imports.add(_node_text(source_bytes, node).strip())

            if node.type in {
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            }:
                type_name = _field_text(source_bytes, node, "name")
                type_owner: str | None
                if type_name:
                    classes.add(type_name)
                    kind = {
                        "class_declaration": "class",
                        "interface_declaration": "interface",
                        "type_alias_declaration": "type",
                        "enum_declaration": "enum",
                    }.get(node.type, "class")
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=type_name,
                        kind=kind,
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    type_owner = _qualified(owner_qualified_name, type_name)
                else:
                    type_owner = owner_qualified_name
                for child in node.children:
                    walk_typescript(child, type_owner)
                return

            if node.type in {
                "function_declaration",
                "method_definition",
                "method_signature",
            }:
                function_name = _field_text(source_bytes, node, "name")
                function_owner: str | None
                if function_name:
                    functions.add(function_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=function_name,
                        kind="method" if owner_qualified_name else "function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    function_owner = _qualified(owner_qualified_name, function_name)
                else:
                    function_owner = owner_qualified_name
                for child in node.children:
                    walk_typescript(child, function_owner)
                return

            for child in node.children:
                walk_typescript(child, owner_qualified_name)

        walk_typescript(tree.root_node)
        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        for node in _walk(tree.root_node):
            if node.type in {"function_declaration", "method_declaration"}:
                name = _field_text(source_bytes, node, "name")
                if name:
                    functions.add(name)
                    owner: str | None = None
                    if node.type == "method_declaration":
                        receiver = node.child_by_field_name("receiver")
                        receiver_text = (
                            _node_text(source_bytes, receiver)
                            if receiver is not None
                            else ""
                        )
                        receiver_match = re.search(
                            r"([A-Za-z_][A-Za-z0-9_]*)\s*\)?\s*$",
                            receiver_text,
                        )
                        owner = receiver_match.group(1) if receiver_match else None
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=name,
                        kind=(
                            "method"
                            if node.type == "method_declaration"
                            else "function"
                        ),
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner,
                        node=node,
                    )
            elif node.type == "type_spec":
                name = _field_text(source_bytes, node, "name")
                if name:
                    classes.add(name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=name,
                        kind="class",
                        language=self.language,
                        source_path=path,
                        node=node,
                    )
            elif node.type in {"import_declaration", "import_spec"}:
                imports.add(_node_text(source_bytes, node).strip())

        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        def walk_rust(node: TSNode, owner_qualified_name: str | None = None) -> None:
            if node.type in {"use_declaration", "extern_crate_declaration"}:
                imports.add(_node_text(source_bytes, node).strip())

            if node.type in {"struct_item", "enum_item", "trait_item"}:
                type_name = _field_text(
                    source_bytes, node, "name"
                ) or _first_identifier(source_bytes, node)
                type_owner: str | None
                if type_name:
                    classes.add(type_name)
                    kind = {
                        "struct_item": "struct",
                        "enum_item": "enum",
                        "trait_item": "trait",
                    }.get(node.type, "class")
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=type_name,
                        kind=kind,
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    type_owner = _qualified(owner_qualified_name, type_name)
                else:
                    type_owner = owner_qualified_name
                for child in node.children:
                    walk_rust(child, type_owner)
                return

            if node.type == "impl_item":
                impl_name = _field_text(
                    source_bytes, node, "name"
                ) or _first_identifier(source_bytes, node)
                impl_owner: str | None
                if impl_name:
                    classes.add(impl_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=impl_name,
                        kind="impl",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    impl_owner = _qualified(owner_qualified_name, impl_name)
                else:
                    impl_owner = owner_qualified_name
                for child in node.children:
                    walk_rust(child, impl_owner)
                return

            if node.type == "function_item":
                function_name = _field_text(source_bytes, node, "name")
                function_owner: str | None
                if function_name:
                    functions.add(function_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=function_name,
                        kind="method" if owner_qualified_name else "function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    function_owner = _qualified(owner_qualified_name, function_name)
                else:
                    function_owner = owner_qualified_name
                for child in node.children:
                    walk_rust(child, function_owner)
                return

            for child in node.children:
                walk_rust(child, owner_qualified_name)

        walk_rust(tree.root_node)
        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


class CSharpAnalyzer(_BaseAnalyzer):
    """Tree-sitter analyzer for C# files."""

    language: SourceLanguage = "csharp"
    parser_name = "csharp"
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
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        tree, source_bytes = self._parse(source_text)

        functions: set[str] = set()
        classes: set[str] = set()
        imports: set[str] = set()
        symbols: list[SymbolSummary] = []
        seen: set[tuple[str, str, str]] = set()

        def walk_csharp(node: TSNode, owner_qualified_name: str | None = None) -> None:
            if node.type in {"using_directive", "extern_alias_directive"}:
                imports.add(_node_text(source_bytes, node).strip())

            if node.type in {
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
            }:
                type_name = _field_text(source_bytes, node, "name")
                type_owner: str | None
                if type_name:
                    classes.add(type_name)
                    kind = {
                        "class_declaration": "class",
                        "interface_declaration": "interface",
                        "struct_declaration": "struct",
                        "enum_declaration": "enum",
                        "record_declaration": "record",
                    }.get(node.type, "class")
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=type_name,
                        kind=kind,
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    type_owner = _qualified(owner_qualified_name, type_name)
                else:
                    type_owner = owner_qualified_name

                for child in node.children:
                    walk_csharp(child, type_owner)
                return

            if node.type in {
                "method_declaration",
                "constructor_declaration",
                "local_function_statement",
            }:
                function_name = _field_text(source_bytes, node, "name")
                function_owner: str | None
                if function_name:
                    functions.add(function_name)
                    _append_symbol(
                        symbols=symbols,
                        seen=seen,
                        name=function_name,
                        kind="method" if owner_qualified_name else "function",
                        language=self.language,
                        source_path=path,
                        owner_qualified_name=owner_qualified_name,
                        node=node,
                    )
                    function_owner = _qualified(owner_qualified_name, function_name)
                else:
                    function_owner = owner_qualified_name

                for child in node.children:
                    walk_csharp(child, function_owner)
                return

            for child in node.children:
                walk_csharp(child, owner_qualified_name)

        walk_csharp(tree.root_node)
        _append_module_symbol(
            symbols=symbols,
            seen=seen,
            path=path,
            language=self.language,
            classes=classes,
            functions=functions,
        )

        summary = CodeUnitSummary(
            path=path,
            language=self.language,
            functions=sorted(functions),
            classes=sorted(classes),
            docstrings=[],
            imports=sorted(item for item in imports if item),
        )
        return summary, sorted(
            symbols,
            key=lambda item: (
                item.kind,
                item.qualified_name or item.name,
                item.line_start or 0,
            ),
        )


class UnknownAnalyzer:
    """Fallback analyzer for unsupported languages."""

    language: SourceLanguage = "unknown"
    supported_suffixes: tuple[str, ...] = ()

    _signature_patterns = (
        re.compile(r"^(def|class|interface|enum|struct|template|fn|func|impl)\b"),
    )
    _dependency_patterns = (
        re.compile(
            r"^(import|from\s+\S+\s+import|#\s*include|include|package|using|use)\b"
        ),
    )

    def supports(self, path: Path) -> bool:
        return False

    def analyze(
        self, path: Path, source_text: str
    ) -> tuple[CodeUnitSummary, list[SymbolSummary]]:
        imports = sorted(
            {
                line.strip()
                for line in source_text.splitlines()
                if any(
                    pattern.match(line.strip()) for pattern in self._dependency_patterns
                )
            }
        )
        return (
            CodeUnitSummary(
                path=path,
                language=self.language,
                functions=[],
                classes=[],
                docstrings=[],
                imports=imports,
            ),
            [],
        )

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
