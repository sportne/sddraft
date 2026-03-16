"""Analysis subsystem APIs."""

from .commit_impact import build_commit_impact
from .evidence_builder import (
    build_generation_evidence_packs,
    build_update_evidence_packs,
)
from .graph_build import build_graph_store
from .graph_index import default_graph_manifest_path, load_graph_store
from .graph_retrieval import (
    GraphExpansionCandidateSource,
    LexicalCandidateSource,
    VectorCandidateSource,
    collect_graph_candidates,
    rerank_evidence,
)
from .hierarchy import (
    build_hierarchy_index,
    default_hierarchy_index_path,
    directory_node_id,
    expand_chunks_with_hierarchy,
    file_node_id,
    hierarchy_chunks,
    iter_hierarchy_chunks,
    load_hierarchy_index,
    save_hierarchy_index,
)
from .metrics import RunMetricsCollector, estimate_rss_mb
from .retrieval import (
    BM25Retriever,
    JsonlChunkSink,
    LexicalIndexer,
    RetrievalQueryEngine,
    ScoredChunk,
    build_document_chunks,
    build_retrieval_store,
    default_retrieval_store_path,
    load_retrieval_index,
    load_retrieval_manifest,
    migrate_legacy_index,
    open_query_engine,
    resolve_retrieval_store_path,
    save_retrieval_index,
    save_retrieval_manifest,
    to_citations,
    tokenize,
)

__all__ = [
    "build_commit_impact",
    "build_generation_evidence_packs",
    "build_update_evidence_packs",
    "build_graph_store",
    "default_graph_manifest_path",
    "load_graph_store",
    "LexicalCandidateSource",
    "GraphExpansionCandidateSource",
    "VectorCandidateSource",
    "collect_graph_candidates",
    "rerank_evidence",
    "file_node_id",
    "directory_node_id",
    "build_hierarchy_index",
    "hierarchy_chunks",
    "iter_hierarchy_chunks",
    "save_hierarchy_index",
    "load_hierarchy_index",
    "default_hierarchy_index_path",
    "expand_chunks_with_hierarchy",
    "RunMetricsCollector",
    "estimate_rss_mb",
    "tokenize",
    "LexicalIndexer",
    "JsonlChunkSink",
    "BM25Retriever",
    "RetrievalQueryEngine",
    "ScoredChunk",
    "build_document_chunks",
    "default_retrieval_store_path",
    "build_retrieval_store",
    "to_citations",
    "save_retrieval_index",
    "load_retrieval_index",
    "save_retrieval_manifest",
    "load_retrieval_manifest",
    "resolve_retrieval_store_path",
    "open_query_engine",
    "migrate_legacy_index",
]
