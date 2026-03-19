"""Graph-aware retrieval expansion and deterministic evidence reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from sddraft.analysis.graph_models import (
    GraphEdgeType,
    GraphStore,
    chunk_node_id,
    file_node_id,
    section_node_id,
)
from sddraft.analysis.retrieval import RetrievalQueryEngine, ScoredChunk, tokenize
from sddraft.domain.models import (
    ChunkInclusionReason,
    GraphInclusionPath,
    KnowledgeChunk,
)

QueryIntent = Literal["implementation", "dependency", "documentation", "architecture"]
TextCandidateSourceName = Literal["lexical", "hierarchy", "vector"]
_LEXICAL_WEIGHT = 0.65
_ANCHOR_WEIGHT = 0.20
_GRAPH_WEIGHT = 0.15
_MAX_GRAPH_PATHS_PER_REASON = 4


@dataclass(frozen=True, slots=True)
class GraphTraversalHit:
    """One traversed graph node hit."""

    node_id: str
    distance: int
    via_edge_type: GraphEdgeType | None
    edge_source_id: str | None
    edge_target_id: str | None


@dataclass(frozen=True, slots=True)
class GraphChunkCandidate:
    """One chunk candidate surfaced by graph expansion."""

    chunk: KnowledgeChunk
    graph_score: float
    reason: str
    graph_paths: tuple[GraphInclusionPath, ...] = ()
    related_files: tuple[Path, ...] = ()
    related_symbols: tuple[str, ...] = ()
    related_sections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnchorSet:
    """Anchors extracted from primary retrieval evidence."""

    node_ids: set[str]
    file_paths: set[Path]
    symbol_ids: set[str]
    symbol_labels: set[str]
    symbol_names: set[str]
    section_ids: set[str]


@dataclass(slots=True)
class RerankResult:
    """Final reranked evidence selection."""

    chunks: list[KnowledgeChunk]
    reasons: list[ChunkInclusionReason]
    related_files: list[Path]
    related_symbols: list[str]
    related_sections: list[str]


@dataclass(frozen=True, slots=True)
class SourceContext:
    """Shared retrieval source context."""

    query: str
    top_k: int
    request_top_k: int
    seed_chunks: list[KnowledgeChunk]
    lexical_scored: list[ScoredChunk]


@dataclass(frozen=True, slots=True)
class TextSourceCandidate:
    """One scored text candidate from lexical/hierarchy/vector sources."""

    source: TextCandidateSourceName
    scored_chunk: ScoredChunk


@dataclass(frozen=True, slots=True)
class GraphSourceResult:
    """Graph source result with candidates and traversal context."""

    source: Literal["graph"]
    candidates: list[GraphChunkCandidate]
    anchors: AnchorSet
    intent: QueryIntent


class LexicalCandidateSource:
    """Primary lexical candidate source."""

    name = "lexical"

    def __init__(self, engine: RetrievalQueryEngine) -> None:
        self._engine = engine

    def collect(
        self,
        context: SourceContext | None = None,
        *,
        query: str | None = None,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Return lexical BM25 candidates for the user query."""
        if context is not None:
            return self._engine.search_scored(context.query, top_k=context.top_k)
        if query is None or top_k is None:
            raise ValueError("query and top_k are required when context is omitted")
        return self._engine.search_scored(query, top_k=top_k)

    def collect_candidates(self, context: SourceContext) -> list[TextSourceCandidate]:
        """Return lexical candidates in the shared text-candidate shape."""

        return [
            TextSourceCandidate(source="lexical", scored_chunk=item)
            for item in self.collect(context)
        ]


class VectorCandidateSource:
    """Vector candidate source placeholder for future implementation."""

    name = "vector"

    def collect(self, *, query: str, top_k: int) -> list[ScoredChunk]:
        """Return no candidates until a real vector backend is implemented."""

        _ = (query, top_k)
        return []

    def collect_candidates(self, context: SourceContext) -> list[TextSourceCandidate]:
        """Return vector candidates in the shared text-candidate shape."""

        _ = context
        return []


class TextCandidateSource(Protocol):
    """Shared contract for lexical/hierarchy/vector text sources."""

    name: str

    def collect_candidates(self, context: SourceContext) -> list[TextSourceCandidate]:
        """Collect deterministic text candidates for this source."""
        ...


class GraphExpansionCandidateSource:
    """Graph-neighborhood candidate source built from lexical seed chunks."""

    name = "graph"

    def __init__(
        self,
        *,
        engine: RetrievalQueryEngine,
        store: GraphStore,
        depth: int,
        top_k: int,
    ) -> None:
        self._engine = engine
        self._store = store
        self._depth = depth
        self._top_k = top_k

    def collect(
        self,
        *,
        query: str,
        seed_chunks: list[KnowledgeChunk],
    ) -> tuple[list[GraphChunkCandidate], AnchorSet, QueryIntent]:
        """Backward-compatible graph candidate collector."""

        return collect_graph_candidates(
            query=query,
            engine=self._engine,
            store=self._store,
            seed_chunks=seed_chunks,
            depth=self._depth,
            top_k=self._top_k,
        )

    def collect_result(self, context: SourceContext) -> GraphSourceResult:
        """Collect graph candidates using shared source context."""

        candidates, anchors, intent = self.collect(
            query=context.query,
            seed_chunks=context.seed_chunks,
        )
        return GraphSourceResult(
            source="graph",
            candidates=candidates,
            anchors=anchors,
            intent=intent,
        )


class HierarchyCandidateSource:
    """Hierarchy-expanded candidates projected into text-source candidates."""

    name = "hierarchy"

    def collect_candidates(self, context: SourceContext) -> list[TextSourceCandidate]:
        """Return hierarchy-expanded seed chunks as scored hierarchy candidates."""

        lexical_ids = {item.chunk.chunk_id for item in context.lexical_scored}
        return [
            TextSourceCandidate(
                source="hierarchy", scored_chunk=ScoredChunk(chunk=item, score=0.0)
            )
            for item in context.seed_chunks
            if item.chunk_id not in lexical_ids
        ]


def collect_text_candidates(
    *,
    sources: list[TextCandidateSource],
    context: SourceContext,
) -> list[TextSourceCandidate]:
    """Collect and deterministically merge text-source candidates."""

    rows: list[TextSourceCandidate] = []
    for source in sources:
        source_rows = source.collect_candidates(context)
        rows.extend(source_rows)
    rows.sort(
        key=lambda item: (
            item.scored_chunk.chunk.chunk_id,
            item.source,
            -item.scored_chunk.score,
        )
    )
    return rows


def infer_query_intent(question: str) -> QueryIntent:
    """Infer deterministic retrieval intent from question keywords."""

    tokens = set(tokenize(question))
    if tokens & {"depend", "dependency", "dependencies", "import", "imports"}:
        return "dependency"
    if tokens & {
        "document",
        "documents",
        "section",
        "sections",
        "grounded",
        "trace",
        "traceability",
        "impact",
        "impacted",
    }:
        return "documentation"
    if tokens & {"architecture", "module", "modules", "design", "overview"}:
        return "architecture"
    return "implementation"


def preferred_edge_types(intent: QueryIntent) -> set[GraphEdgeType]:
    """Return edge-type filters for one intent."""

    if intent == "dependency":
        return {"imports", "contains", "defines", "references"}
    if intent == "documentation":
        return {"documents", "impacts_section", "contains", "references"}
    if intent == "architecture":
        return {
            "contains",
            "parent_of",
            "defines",
            "references",
            "documents",
            "imports",
        }
    return {"defines", "contains", "references", "parent_of"}


def _tie_break_key(chunk: KnowledgeChunk) -> tuple[str, int, str]:
    return (chunk.source_path.as_posix(), chunk.line_start or 0, chunk.chunk_id)


def _iter_neighbor_edges(store: GraphStore, node_id: str) -> list[tuple[str, str]]:
    edge_ids = store.outgoing.get(node_id, []) + store.incoming.get(node_id, [])
    values: list[tuple[str, str]] = []
    for edge_id in sorted(set(edge_ids)):
        edge = store.edges_by_id.get(edge_id)
        if edge is None:
            continue
        neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
        values.append((edge_id, neighbor))
    return values


def _symbol_label(store: GraphStore, symbol_id: str) -> str | None:
    record = store.symbol_records.get(symbol_id)
    if record is not None:
        name = record.qualified_name or record.name
        return f"{name} [{record.file_path.as_posix()}]"

    node = store.nodes_by_id.get(symbol_id)
    if node is None or node.node_type != "symbol":
        return None
    name = node.qualified_name or node.symbol_name or node.label
    if node.path is not None:
        return f"{name} [{node.path.as_posix()}]"
    return name


def _symbol_ids_for_path(store: GraphStore, path: Path) -> set[str]:
    values = {
        record.symbol_id
        for record in store.symbol_records.values()
        if record.file_path == path
    }
    if values:
        return values
    return {
        node_id
        for node_id, node in store.nodes_by_id.items()
        if node.node_type == "symbol" and node.path == path
    }


def _collect_anchor_symbol_ids(store: GraphStore, node_ids: set[str]) -> set[str]:
    symbol_ids: set[str] = set()
    related_file_paths: set[Path] = set()
    for node_id in sorted(node_ids):
        node = store.nodes_by_id.get(node_id)
        if node is None:
            continue
        if node.node_type == "symbol":
            symbol_ids.add(node.node_id)
            continue
        if node.node_type == "file" and node.path is not None:
            related_file_paths.add(node.path)

        for edge_id, neighbor_id in _iter_neighbor_edges(store, node_id):
            edge = store.edges_by_id.get(edge_id)
            neighbor = store.nodes_by_id.get(neighbor_id)
            if edge is None or neighbor is None:
                continue
            if edge.edge_type not in {
                "references",
                "defines",
                "contains",
                "documents",
                "impacts_section",
            }:
                continue
            if neighbor.node_type == "symbol":
                symbol_ids.add(neighbor.node_id)
            elif neighbor.node_type == "file" and neighbor.path is not None:
                related_file_paths.add(neighbor.path)

    for path in sorted(related_file_paths, key=lambda item: item.as_posix()):
        symbol_ids.update(_symbol_ids_for_path(store, path))

    return symbol_ids


def extract_anchors(chunks: list[KnowledgeChunk], store: GraphStore) -> AnchorSet:
    """Extract anchor entities from current evidence chunks."""

    node_ids: set[str] = set()
    file_paths: set[Path] = set()
    symbol_ids: set[str] = set()
    symbol_labels: set[str] = set()
    symbol_names: set[str] = set()
    section_ids: set[str] = set()

    for chunk in chunks:
        chunk_node = chunk_node_id(chunk.chunk_id)
        if chunk_node in store.nodes_by_id:
            node_ids.add(chunk_node)

        file_node = file_node_id(chunk.source_path)
        if file_node in store.nodes_by_id:
            node_ids.add(file_node)
            file_paths.add(chunk.source_path)

        if chunk.section_id:
            sid = section_node_id(chunk.section_id)
            if sid in store.nodes_by_id:
                node_ids.add(sid)
                section_ids.add(chunk.section_id)

    symbol_ids.update(_collect_anchor_symbol_ids(store, node_ids))
    for path in sorted(file_paths, key=lambda item: item.as_posix()):
        symbol_ids.update(_symbol_ids_for_path(store, path))
    node_ids.update(symbol_ids)

    for symbol_id in sorted(symbol_ids):
        label = _symbol_label(store, symbol_id)
        if label is None:
            continue
        symbol_labels.add(label)
        symbol_names.add(label.split("[", maxsplit=1)[0].strip().split(".")[-1].lower())

    return AnchorSet(
        node_ids=node_ids,
        file_paths=file_paths,
        symbol_ids=symbol_ids,
        symbol_labels=symbol_labels,
        symbol_names=symbol_names,
        section_ids=section_ids,
    )


def expand_graph_neighbors(
    *,
    store: GraphStore,
    anchors: AnchorSet,
    depth: int,
    edge_filter: set[GraphEdgeType],
    limit: int,
) -> list[GraphTraversalHit]:
    """Expand graph neighborhood from anchor nodes with bounded BFS."""

    if not anchors.node_ids or depth <= 0 or limit <= 0:
        return []

    frontier = sorted(anchors.node_ids)
    visited = set(frontier)
    hits: list[GraphTraversalHit] = []

    for distance in range(1, depth + 1):
        next_frontier: list[str] = []
        for node_id in frontier:
            edge_ids = store.outgoing.get(node_id, []) + store.incoming.get(node_id, [])
            for edge_id in sorted(set(edge_ids)):
                edge = store.edges_by_id.get(edge_id)
                if edge is None or edge.edge_type not in edge_filter:
                    continue
                neighbor = (
                    edge.target_id if edge.source_id == node_id else edge.source_id
                )
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(neighbor)
                hits.append(
                    GraphTraversalHit(
                        node_id=neighbor,
                        distance=distance,
                        via_edge_type=edge.edge_type,
                        edge_source_id=edge.source_id,
                        edge_target_id=edge.target_id,
                    )
                )
                if len(hits) >= limit:
                    return hits
        frontier = sorted(set(next_frontier))
        if not frontier:
            break

    return hits


def _load_graph_candidate_chunks(
    *,
    engine: RetrievalQueryEngine,
    store: GraphStore,
    hits: list[GraphTraversalHit],
    top_k: int,
) -> list[GraphChunkCandidate]:
    if not hits:
        return []

    def path_from_hit(hit: GraphTraversalHit) -> GraphInclusionPath | None:
        if (
            hit.via_edge_type is None
            or hit.edge_source_id is None
            or hit.edge_target_id is None
        ):
            return None
        source_node = store.nodes_by_id.get(hit.edge_source_id)
        target_node = store.nodes_by_id.get(hit.edge_target_id)
        return GraphInclusionPath(
            distance=max(hit.distance, 1),
            edge_type=hit.via_edge_type,
            source_node_id=hit.edge_source_id,
            source_node_type=source_node.node_type if source_node is not None else None,
            source_label=source_node.label if source_node is not None else None,
            target_node_id=hit.edge_target_id,
            target_node_type=target_node.node_type if target_node is not None else None,
            target_label=target_node.label if target_node is not None else None,
        )

    chunk_ids: set[str] = set()
    source_paths: set[Path] = set()
    section_ids: set[str] = set()
    symbol_hits: list[tuple[GraphTraversalHit, GraphInclusionPath | None]] = []

    for hit in hits:
        node = store.nodes_by_id.get(hit.node_id)
        if node is None:
            continue
        if node.node_type == "chunk" and node.chunk_id:
            chunk_ids.add(node.chunk_id)
        elif node.node_type == "file" and node.path is not None:
            source_paths.add(node.path)
        elif node.node_type == "sdd_section" and node.section_id:
            section_ids.add(node.section_id)
        elif node.node_type == "symbol":
            symbol_hits.append((hit, path_from_hit(hit)))
            if node.path is not None:
                source_paths.add(node.path)

    loaded: dict[str, KnowledgeChunk] = {}

    for chunk in engine.load_chunks_by_chunk_ids(chunk_ids, limit=top_k * 2):
        loaded[chunk.chunk_id] = chunk

    for chunk in engine.load_chunks_by_source_paths(source_paths, limit=top_k * 4):
        loaded.setdefault(chunk.chunk_id, chunk)

    for chunk in engine.load_chunks_by_section_ids(section_ids, limit=top_k * 2):
        loaded.setdefault(chunk.chunk_id, chunk)

    scored: dict[str, GraphChunkCandidate] = {}

    def add_candidate(
        *,
        chunk: KnowledgeChunk,
        score: float,
        reason: str,
        related_files: set[Path] | None = None,
        related_symbols: set[str] | None = None,
        related_sections: set[str] | None = None,
        graph_paths: list[GraphInclusionPath] | None = None,
    ) -> None:
        current = scored.get(chunk.chunk_id)
        path_by_key: dict[tuple[int, str, str, str], GraphInclusionPath] = {}
        for item in (current.graph_paths if current is not None else tuple()):
            key = (
                item.distance,
                item.edge_type,
                item.source_node_id,
                item.target_node_id,
            )
            path_by_key[key] = item
        for item in graph_paths or []:
            key = (
                item.distance,
                item.edge_type,
                item.source_node_id,
                item.target_node_id,
            )
            path_by_key[key] = item
        new_paths = tuple(
            item
            for _, item in sorted(
                path_by_key.items(),
                key=lambda pair: pair[0],
            )
        )
        new_files = tuple(
            sorted(
                {
                    *(current.related_files if current is not None else tuple()),
                    *(related_files or set()),
                },
                key=lambda item: item.as_posix(),
            )
        )
        new_symbols = tuple(
            sorted(
                {
                    *(current.related_symbols if current is not None else tuple()),
                    *(related_symbols or set()),
                }
            )
        )
        new_sections = tuple(
            sorted(
                {
                    *(current.related_sections if current is not None else tuple()),
                    *(related_sections or set()),
                }
            )
        )
        preferred_reason = reason
        preferred_score = score
        if current is not None:
            if current.graph_score > score:
                preferred_score = current.graph_score
                preferred_reason = current.reason
            elif current.graph_score == score:
                preferred_reason = min(current.reason, reason)
        scored[chunk.chunk_id] = GraphChunkCandidate(
            chunk=chunk,
            graph_score=preferred_score,
            reason=preferred_reason,
            graph_paths=new_paths,
            related_files=new_files,
            related_symbols=new_symbols,
            related_sections=new_sections,
        )

    def symbols_for_path(path: Path) -> set[str]:
        values: set[str] = set()
        for symbol_id in _symbol_ids_for_path(store, path):
            label = _symbol_label(store, symbol_id)
            if label is not None:
                values.add(label)
        return values

    for hit in hits:
        node = store.nodes_by_id.get(hit.node_id)
        if node is None:
            continue

        reason = f"graph:{node.node_type}:{node.label}"
        score = 1.0 / max(hit.distance, 1)
        hit_path = path_from_hit(hit)
        graph_paths = [hit_path] if hit_path is not None else []

        if node.node_type == "chunk" and node.chunk_id and node.chunk_id in loaded:
            chunk = loaded[node.chunk_id]
            add_candidate(
                chunk=chunk,
                score=score,
                reason=reason,
                related_files={chunk.source_path},
                related_sections={chunk.section_id} if chunk.section_id else set(),
                graph_paths=graph_paths,
            )
            continue

        if node.node_type == "file" and node.path is not None:
            symbol_labels = symbols_for_path(node.path)
            for chunk in loaded.values():
                if chunk.source_path == node.path:
                    add_candidate(
                        chunk=chunk,
                        score=score,
                        reason=reason,
                        related_files={node.path},
                        related_symbols=symbol_labels,
                        related_sections=(
                            {chunk.section_id} if chunk.section_id else set()
                        ),
                        graph_paths=graph_paths,
                    )
            continue

        if node.node_type == "sdd_section" and node.section_id:
            for chunk in loaded.values():
                if chunk.section_id == node.section_id:
                    add_candidate(
                        chunk=chunk,
                        score=score,
                        reason=reason,
                        related_files={chunk.source_path},
                        related_sections={node.section_id},
                        graph_paths=graph_paths,
                    )
            continue

    for hit, symbol_path in symbol_hits:
        node = store.nodes_by_id.get(hit.node_id)
        if node is None or node.path is None:
            continue
        score = 1.0 / max(hit.distance, 1)
        reason = f"graph:symbol:{node.label}"
        symbol_label = _symbol_label(store, node.node_id)
        related_symbol_labels = {symbol_label} if symbol_label is not None else set()
        graph_paths = [symbol_path] if symbol_path is not None else []
        for chunk in loaded.values():
            if chunk.source_path != node.path:
                continue
            if (
                node.line_start is not None
                and chunk.line_start is not None
                and chunk.line_end is not None
            ):
                if chunk.line_start <= node.line_start <= chunk.line_end:
                    add_candidate(
                        chunk=chunk,
                        score=score,
                        reason=reason,
                        related_files={node.path},
                        related_symbols=related_symbol_labels,
                        related_sections=(
                            {chunk.section_id} if chunk.section_id else set()
                        ),
                        graph_paths=graph_paths,
                    )
            elif node.symbol_name and node.symbol_name.lower() in tokenize(chunk.text):
                add_candidate(
                    chunk=chunk,
                    score=score,
                    reason=reason,
                    related_files={node.path},
                    related_symbols=related_symbol_labels,
                    related_sections={chunk.section_id} if chunk.section_id else set(),
                    graph_paths=graph_paths,
                )

    ranked = sorted(
        scored.values(),
        key=lambda item: (-item.graph_score, *_tie_break_key(item.chunk)),
    )
    return ranked[: max(top_k, 1)]


def _type_bias(chunk: KnowledgeChunk, intent: QueryIntent) -> float:
    if intent == "dependency":
        if chunk.source_type == "code":
            return 0.06
        if chunk.source_type in {"file_summary", "directory_summary"}:
            return 0.03
        return 0.01

    if intent == "documentation":
        if chunk.source_type == "sdd_section":
            return 0.07
        if chunk.source_type == "review_artifact":
            return 0.04
        return 0.01

    if intent == "architecture":
        if chunk.source_type in {"directory_summary", "file_summary"}:
            return 0.06
        if chunk.source_type == "code":
            return 0.02
        return 0.01

    # implementation
    if chunk.source_type == "code":
        return 0.07
    if chunk.source_type == "file_summary":
        return 0.03
    return 0.01


def rerank_evidence(
    *,
    lexical_candidates: list[ScoredChunk],
    graph_candidates: list[GraphChunkCandidate],
    anchors: AnchorSet,
    intent: QueryIntent,
    top_k: int,
) -> RerankResult:
    """Combine lexical and graph candidates into deterministic ranked evidence."""

    if top_k <= 0:
        return RerankResult([], [], [], [], [])

    # Normalize lexical scores to a 0-1 range so they can be blended with
    # graph proximity and anchor signals in a transparent formula.
    lexical_max = max((item.score for item in lexical_candidates), default=0.0)
    lexical_by_chunk_id = {item.chunk.chunk_id: item for item in lexical_candidates}
    graph_by_chunk_id = {item.chunk.chunk_id: item for item in graph_candidates}

    candidate_chunks: dict[str, KnowledgeChunk] = {}
    for lexical_seed in lexical_candidates:
        candidate_chunks[lexical_seed.chunk.chunk_id] = lexical_seed.chunk
    for graph_seed in graph_candidates:
        candidate_chunks.setdefault(graph_seed.chunk.chunk_id, graph_seed.chunk)

    ranked_rows: list[tuple[float, KnowledgeChunk, ChunkInclusionReason]] = []

    for chunk_id, chunk in sorted(candidate_chunks.items()):
        lexical_item = lexical_by_chunk_id.get(chunk_id)
        lexical_raw = lexical_item.score if lexical_item is not None else 0.0
        lexical_norm = (lexical_raw / lexical_max) if lexical_max > 0 else 0.0

        anchor_score = 0.0
        if chunk.source_path in anchors.file_paths:
            anchor_score = max(anchor_score, 1.0)
        if chunk.section_id and chunk.section_id in anchors.section_ids:
            anchor_score = max(anchor_score, 1.0)
        if anchors.symbol_names and anchors.symbol_names.intersection(
            set(tokenize(chunk.text))
        ):
            anchor_score = max(anchor_score, 0.8)

        graph_item = graph_by_chunk_id.get(chunk_id)
        graph_score = graph_item.graph_score if graph_item is not None else 0.0
        type_bias = _type_bias(chunk, intent)
        # Weighted score tuned for v1:
        # lexical relevance is primary, graph/anchor add contextual lift.
        final_score = (
            _LEXICAL_WEIGHT * lexical_norm
            + _ANCHOR_WEIGHT * anchor_score
            + _GRAPH_WEIGHT * graph_score
            + type_bias
        )

        source: Literal["lexical", "hierarchy", "graph", "vector"] = "lexical"
        reason_text = "lexical"
        if graph_item is not None and lexical_item is None:
            source = "graph"
            reason_text = graph_item.reason
        elif graph_item is not None and lexical_item is not None:
            source = "graph"
            reason_text = f"lexical+{graph_item.reason}"

        reason = ChunkInclusionReason(
            chunk_id=chunk_id,
            source=source,
            lexical_score=round(lexical_norm, 6),
            anchor_score=round(anchor_score, 6),
            graph_score=round(graph_score, 6),
            type_bias=round(type_bias, 6),
            final_score=round(final_score, 6),
            reason=reason_text,
            graph_paths=(
                list(graph_item.graph_paths[:_MAX_GRAPH_PATHS_PER_REASON])
                if graph_item is not None
                else []
            ),
        )
        ranked_rows.append((final_score, chunk, reason))

    ranked_rows.sort(
        key=lambda item: (-item[0], *_tie_break_key(item[1])),
    )

    selected = ranked_rows[:top_k]
    selected_chunks = [item[1] for item in selected]
    selected_reasons = [item[2] for item in selected]

    related_files: set[Path] = set()
    related_symbols: set[str] = set()
    related_sections: set[str] = set()

    # Related entities are supplemental context shown to the prompt/response.
    # They are kept separate from primary selected chunks.
    for candidate in graph_candidates:
        for path in candidate.related_files:
            if path not in anchors.file_paths:
                related_files.add(path)
        for section_id in candidate.related_sections:
            if section_id not in anchors.section_ids:
                related_sections.add(section_id)
        for label in candidate.related_symbols:
            related_symbols.add(label)

    return RerankResult(
        chunks=selected_chunks,
        reasons=selected_reasons,
        related_files=sorted(related_files, key=lambda item: item.as_posix()),
        related_symbols=sorted(related_symbols),
        related_sections=sorted(related_sections),
    )


def collect_graph_candidates(
    *,
    query: str,
    engine: RetrievalQueryEngine,
    store: GraphStore,
    seed_chunks: list[KnowledgeChunk],
    depth: int,
    top_k: int,
) -> tuple[list[GraphChunkCandidate], AnchorSet, QueryIntent]:
    """Collect graph-driven chunk candidates from seed evidence."""

    intent = infer_query_intent(query)
    anchors = extract_anchors(seed_chunks, store)
    hits = expand_graph_neighbors(
        store=store,
        anchors=anchors,
        depth=max(1, min(depth, 2)),
        edge_filter=preferred_edge_types(intent),
        limit=max(top_k * 4, 8),
    )
    candidates = _load_graph_candidate_chunks(
        engine=engine,
        store=store,
        hits=hits,
        top_k=max(top_k * 2, 6),
    )
    return candidates, anchors, intent


def flatten_text_candidates(
    candidates: list[TextSourceCandidate],
) -> tuple[list[ScoredChunk], list[ScoredChunk], list[ScoredChunk]]:
    """Split merged text candidates into lexical/hierarchy/vector scored chunks."""

    lexical: list[ScoredChunk] = []
    hierarchy: list[ScoredChunk] = []
    vector: list[ScoredChunk] = []
    for candidate in candidates:
        if candidate.source == "lexical":
            lexical.append(candidate.scored_chunk)
        elif candidate.source == "hierarchy":
            hierarchy.append(candidate.scored_chunk)
        else:
            vector.append(candidate.scored_chunk)
    lexical.sort(key=lambda item: item.chunk.chunk_id)
    hierarchy.sort(key=lambda item: item.chunk.chunk_id)
    vector.sort(key=lambda item: item.chunk.chunk_id)
    seen_lexical: set[str] = set()
    deduped_lexical: list[ScoredChunk] = []
    for scored_item in lexical:
        if scored_item.chunk.chunk_id in seen_lexical:
            continue
        seen_lexical.add(scored_item.chunk.chunk_id)
        deduped_lexical.append(scored_item)
    seen_hierarchy = seen_lexical.copy()
    deduped_hierarchy: list[ScoredChunk] = []
    for scored_item in hierarchy:
        if scored_item.chunk.chunk_id in seen_hierarchy:
            continue
        seen_hierarchy.add(scored_item.chunk.chunk_id)
        deduped_hierarchy.append(scored_item)
    seen_vector = seen_hierarchy.copy()
    deduped_vector: list[ScoredChunk] = []
    for scored_item in vector:
        if scored_item.chunk.chunk_id in seen_vector:
            continue
        seen_vector.add(scored_item.chunk.chunk_id)
        deduped_vector.append(scored_item)
    return deduped_lexical, deduped_hierarchy, deduped_vector
