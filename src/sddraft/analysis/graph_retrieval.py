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
from sddraft.domain.models import ChunkInclusionReason, KnowledgeChunk

QueryIntent = Literal["implementation", "dependency", "documentation", "architecture"]


@dataclass(frozen=True, slots=True)
class GraphTraversalHit:
    """One traversed graph node hit."""

    node_id: str
    distance: int
    via_edge_type: GraphEdgeType | None


@dataclass(frozen=True, slots=True)
class GraphChunkCandidate:
    """One chunk candidate surfaced by graph expansion."""

    chunk: KnowledgeChunk
    graph_score: float
    reason: str


@dataclass(frozen=True, slots=True)
class AnchorSet:
    """Anchors extracted from primary retrieval evidence."""

    node_ids: set[str]
    file_paths: set[Path]
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


class CandidateSource(Protocol):
    """Pluggable candidate source (lexical, graph, vector placeholder)."""

    name: str

    def collect(self, *, query: str, top_k: int) -> list[ScoredChunk]:
        """Collect scored retrieval candidates."""
        ...


class LexicalCandidateSource:
    """Primary lexical candidate source."""

    name = "lexical"

    def __init__(self, engine: RetrievalQueryEngine) -> None:
        self._engine = engine

    def collect(self, *, query: str, top_k: int) -> list[ScoredChunk]:
        return self._engine.search_scored(query, top_k=top_k)


class VectorCandidateSource:
    """Vector candidate source placeholder for future implementation."""

    name = "vector"

    def collect(self, *, query: str, top_k: int) -> list[ScoredChunk]:
        _ = (query, top_k)
        return []


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
        return collect_graph_candidates(
            query=query,
            engine=self._engine,
            store=self._store,
            seed_chunks=seed_chunks,
            depth=self._depth,
            top_k=self._top_k,
        )


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


def extract_anchors(chunks: list[KnowledgeChunk], store: GraphStore) -> AnchorSet:
    """Extract anchor entities from current evidence chunks."""

    node_ids: set[str] = set()
    file_paths: set[Path] = set()
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

        for token in tokenize(chunk.text):
            if token in store.symbols_by_name:
                symbol_names.add(token)
                node_ids.update(store.symbols_by_name[token])

    return AnchorSet(
        node_ids=node_ids,
        file_paths=file_paths,
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

    chunk_ids: set[str] = set()
    source_paths: set[Path] = set()
    section_ids: set[str] = set()
    symbol_hits: list[GraphTraversalHit] = []

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
            symbol_hits.append(hit)
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

    for hit in hits:
        node = store.nodes_by_id.get(hit.node_id)
        if node is None:
            continue

        reason = f"graph:{node.node_type}:{node.label}"
        score = 1.0 / max(hit.distance, 1)

        if node.node_type == "chunk" and node.chunk_id and node.chunk_id in loaded:
            chunk = loaded[node.chunk_id]
            scored[chunk.chunk_id] = GraphChunkCandidate(
                chunk=chunk,
                graph_score=score,
                reason=reason,
            )
            continue

        if node.node_type == "file" and node.path is not None:
            for chunk in loaded.values():
                if chunk.source_path == node.path:
                    current = scored.get(chunk.chunk_id)
                    if current is None or score > current.graph_score:
                        scored[chunk.chunk_id] = GraphChunkCandidate(
                            chunk=chunk,
                            graph_score=score,
                            reason=reason,
                        )
            continue

        if node.node_type == "sdd_section" and node.section_id:
            for chunk in loaded.values():
                if chunk.section_id == node.section_id:
                    current = scored.get(chunk.chunk_id)
                    if current is None or score > current.graph_score:
                        scored[chunk.chunk_id] = GraphChunkCandidate(
                            chunk=chunk,
                            graph_score=score,
                            reason=reason,
                        )
            continue

    for hit in symbol_hits:
        node = store.nodes_by_id.get(hit.node_id)
        if node is None or node.path is None:
            continue
        score = 1.0 / max(hit.distance, 1)
        reason = f"graph:symbol:{node.label}"
        for chunk in loaded.values():
            if chunk.source_path != node.path:
                continue
            if (
                node.line_start is not None
                and chunk.line_start is not None
                and chunk.line_end is not None
            ):
                if chunk.line_start <= node.line_start <= chunk.line_end:
                    current = scored.get(chunk.chunk_id)
                    if current is None or score > current.graph_score:
                        scored[chunk.chunk_id] = GraphChunkCandidate(
                            chunk=chunk,
                            graph_score=score,
                            reason=reason,
                        )
            elif node.symbol_name and node.symbol_name.lower() in tokenize(chunk.text):
                current = scored.get(chunk.chunk_id)
                if current is None or score > current.graph_score:
                    scored[chunk.chunk_id] = GraphChunkCandidate(
                        chunk=chunk,
                        graph_score=score,
                        reason=reason,
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
        final_score = (
            0.65 * lexical_norm + 0.20 * anchor_score + 0.15 * graph_score + type_bias
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

    for candidate in graph_candidates:
        chunk = candidate.chunk
        if chunk.source_path not in anchors.file_paths:
            related_files.add(chunk.source_path)
        if chunk.section_id and chunk.section_id not in anchors.section_ids:
            related_sections.add(chunk.section_id)
        for token in tokenize(chunk.text):
            if len(token) <= 2 or token in anchors.symbol_names or token.isdigit():
                continue
            related_symbols.add(token)

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
