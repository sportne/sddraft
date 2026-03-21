"""Typed domain models for the SDDraft pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainModel(BaseModel):
    """Base model with strict shape checking."""

    model_config = ConfigDict(extra="forbid")


SourceLanguage = Literal[
    "python",
    "java",
    "cpp",
    "javascript",
    "typescript",
    "go",
    "rust",
    "csharp",
    "unknown",
]

AskMode = Literal["standard", "intensive"]


class SourcesConfig(DomainModel):
    """Source selection configuration."""

    roots: list[Path] = Field(default_factory=list)
    include: list[str] = Field(
        default_factory=lambda: [
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
        ]
    )
    exclude: list[str] = Field(default_factory=list)


class LLMConfig(DomainModel):
    """LLM provider configuration."""

    provider: str = "mock"
    model_name: str = "mock-sddraft"
    temperature: float = 0.2

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Ensure model temperature stays in the supported 0.0-1.0 range."""

        if value < 0.0 or value > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return value


class GenerationOptions(DomainModel):
    """Optional generation tuning knobs."""

    max_files: int = 500
    code_chunk_lines: int = 40
    retrieval_top_k: int = 6
    write_batch_size: int = 200
    max_in_memory_records: int = 2000
    index_shard_size: int = 1000
    graph_enabled: bool = True
    graph_depth: Literal[1, 2] = 1
    graph_top_k: int = 12
    vector_enabled: bool = False
    vector_top_k: int = 8
    ask_mode_default: AskMode = "standard"
    intensive_chunk_tokens: int = 8192
    intensive_max_selected_excerpts: int = 12

    @field_validator(
        "max_files",
        "code_chunk_lines",
        "retrieval_top_k",
        "write_batch_size",
        "max_in_memory_records",
        "index_shard_size",
        "graph_top_k",
        "vector_top_k",
        "intensive_chunk_tokens",
        "intensive_max_selected_excerpts",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        """Reject zero/negative generation tuning values."""

        if value <= 0:
            raise ValueError("generation options must be positive integers")
        return value


class ProjectConfig(DomainModel):
    """Top-level project configuration."""

    project_name: str
    sources: SourcesConfig
    sdd_template: Path
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    output_dir: Path = Path("artifacts")


class ConfigBundle(DomainModel):
    """Validated config objects used by workflows."""

    project: ProjectConfig
    csc_descriptors: list[CSCDescriptor]
    template: SDDTemplate


class CSCDescriptor(DomainModel):
    """Component descriptor used for SDD generation."""

    csc_id: str
    title: str
    purpose: str
    source_roots: list[Path] = Field(default_factory=list)
    key_files: list[Path] = Field(default_factory=list)
    provided_interfaces: list[str] = Field(default_factory=list)
    used_interfaces: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)


class SDDSectionSpec(DomainModel):
    """Specification for one SDD section."""

    id: str
    title: str
    instruction: str
    evidence_kinds: list[str] = Field(default_factory=list)


class SDDTemplate(DomainModel):
    """SDD document template."""

    document_type: Literal["sdd"] = "sdd"
    sections: list[SDDSectionSpec]


class CodeUnitSummary(DomainModel):
    """Structured summary of one source file."""

    path: Path
    language: SourceLanguage
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    docstrings: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)


class SymbolSummary(DomainModel):
    """Deterministic symbol summary for one source declaration."""

    name: str
    qualified_name: str | None = None
    kind: str
    language: SourceLanguage
    source_path: Path
    owner_qualified_name: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class FileDiffSummary(DomainModel):
    """Normalized file-level diff summary."""

    path: Path
    language: SourceLanguage = "unknown"
    added_lines: int = 0
    removed_lines: int = 0
    signature_changes: list[str] = Field(default_factory=list)
    dependency_changes: list[str] = Field(default_factory=list)
    comment_only: bool = False


class CommitImpact(DomainModel):
    """Higher-level impact classification for a commit range."""

    commit_range: str
    changed_files: list[FileDiffSummary]
    change_kinds: list[str] = Field(default_factory=list)
    impacted_sections: list[str] = Field(default_factory=list)
    summary: str


class EvidenceReference(DomainModel):
    """Traceable evidence pointer used by generated outputs."""

    kind: str = Field(
        description=(
            "Evidence category identifier, such as code_summary, symbol, "
            "dependency, commit_impact, hierarchy_summary, or existing_sdd."
        )
    )
    source: str = Field(
        description=(
            "Deterministic source pointer, usually a relative path or stable "
            "artifact identifier."
        )
    )
    detail: str | None = Field(
        default=None,
        description=(
            "Optional disambiguating detail within source, for example an "
            "interface name, symbol name, section ID, or line hint."
        ),
    )


class SectionEvidencePack(DomainModel):
    """Deterministic section-scoped evidence payload."""

    section: SDDSectionSpec
    csc: CSCDescriptor
    code_summaries: list[CodeUnitSummary] = Field(default_factory=list)
    symbol_summaries: list[SymbolSummary] = Field(default_factory=list)
    dependency_summaries: list[str] = Field(default_factory=list)
    hierarchy_summaries: list[str] = Field(default_factory=list)
    commit_impact: CommitImpact | None = None
    existing_section_text: str | None = None
    evidence_references: list[EvidenceReference] = Field(default_factory=list)


class SectionDraft(DomainModel):
    """Structured generated section draft."""

    section_id: str = Field(
        description="Template section identifier this draft corresponds to."
    )
    title: str = Field(description="Human-readable section title.")
    content: str = Field(
        description=(
            "Draft section content grounded only in supplied evidence. Unknown facts "
            "must be marked as TBD."
        )
    )
    evidence_refs: list[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence references that directly support claims in content.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description=(
            "Assumptions made due to partial evidence. Leave empty when none."
        ),
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Missing inputs needed for a complete section. Use explicit TBD entries "
            "instead of inventing facts."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence that the section content is supported by available evidence "
            "(0.0 low, 1.0 high)."
        ),
    )


class SectionUpdateProposal(DomainModel):
    """Structured section update proposal."""

    section_id: str = Field(
        description="Identifier of the section that should be revised."
    )
    title: str = Field(description="Section title.")
    existing_text: str = Field(
        description="Current section text being reviewed for update."
    )
    proposed_text: str = Field(
        description="Proposed revised section text grounded in supplied evidence."
    )
    rationale: str = Field(
        description=(
            "Reason the update is needed and how evidence supports the proposed text."
        )
    )
    uncertainty_list: list[str] = Field(
        default_factory=list,
        description=(
            "Open questions, risks, or unknowns reviewers should verify. Use TBD "
            "items when details are missing."
        ),
    )
    review_priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Suggested review urgency based on impact and uncertainty.",
    )
    evidence_refs: list[EvidenceReference] = Field(
        default_factory=list,
        description="Evidence references that justify the proposal and rationale.",
    )


class SDDDocument(DomainModel):
    """Composed SDD document from generated section drafts."""

    csc_id: str
    title: str
    sections: list[SectionDraft]


class SectionReviewArtifact(DomainModel):
    """Reviewable metadata for one generated section."""

    section_id: str
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class ReviewArtifact(DomainModel):
    """Review artifact generated with the SDD output."""

    csc_id: str
    sections: list[SectionReviewArtifact]


class UpdateProposalReport(DomainModel):
    """High-level report for impacted section updates."""

    commit_range: str
    impacted_sections: list[str]
    proposals: list[SectionUpdateProposal]


class FileSummaryDoc(DomainModel):
    """Generated summary for one source file in the hierarchy view."""

    node_id: str
    path: Path
    language: SourceLanguage
    summary: str
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the file summary based on supplied evidence.",
    )


class DirectorySummaryDoc(DomainModel):
    """Generated summary for one directory in the hierarchy view."""

    node_id: str
    path: Path
    summary: str
    local_files: list[Path] = Field(default_factory=list)
    child_directories: list[Path] = Field(default_factory=list)
    subtree_rollup: SubtreeRollup = Field(default_factory=lambda: SubtreeRollup())
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the directory subtree summary based on supplied local "
            "evidence and descendant rollups."
        ),
    )


class HierarchyDocArtifact(DomainModel):
    """Structured hierarchy documentation artifact."""

    csc_id: str
    root: Path
    file_summaries: list[FileSummaryDoc] = Field(default_factory=list)
    directory_summaries: list[DirectorySummaryDoc] = Field(default_factory=list)


class FileSummaryRecord(DomainModel):
    """Streamed file-summary record persisted to hierarchy store."""

    node_id: str
    path: Path
    language: SourceLanguage
    summary: str
    functions: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the generated file summary.",
    )


class DirectorySummaryRecord(DomainModel):
    """Streamed directory-summary record persisted to hierarchy store."""

    node_id: str
    path: Path
    summary: str
    local_files: list[Path] = Field(default_factory=list)
    child_directories: list[Path] = Field(default_factory=list)
    subtree_rollup: SubtreeRollup = Field(default_factory=lambda: SubtreeRollup())
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the generated directory summary.",
    )


class SubtreeRollup(DomainModel):
    """Deterministic recursive metadata for one directory subtree."""

    descendant_file_count: int = 0
    descendant_directory_count: int = 0
    language_counts: dict[str, int] = Field(default_factory=dict)
    key_topics: list[str] = Field(default_factory=list)
    representative_files: list[Path] = Field(default_factory=list)


class HierarchyNodeRecord(DomainModel):
    """Node record for hierarchy graph storage."""

    node_id: str
    kind: Literal["file", "directory"]
    path: Path
    parent_id: str | None = None
    doc_path: Path
    abstract: str
    keywords: list[str] = Field(default_factory=list)


class HierarchyEdgeRecord(DomainModel):
    """Edge record for hierarchy graph storage."""

    parent_id: str
    child_id: str
    relation: Literal["contains"] = "contains"


class HierarchyManifest(DomainModel):
    """Hierarchy store manifest for streamed JSONL artifacts."""

    csc_id: str
    root: Path
    version: Literal["v2-stream-jsonl", "v3-stream-jsonl"] = "v3-stream-jsonl"
    file_summaries_path: Path
    directory_summaries_path: Path
    nodes_path: Path
    edges_path: Path


class HierarchyIndexNode(DomainModel):
    """Node entry in hierarchy index."""

    node_id: str
    kind: Literal["file", "directory"]
    path: Path
    parent_id: str | None = None
    doc_path: Path
    abstract: str
    keywords: list[str] = Field(default_factory=list)


class HierarchyIndexEdge(DomainModel):
    """Directed hierarchy relation entry."""

    parent_id: str
    child_id: str
    relation: Literal["contains"] = "contains"


class HierarchyIndex(DomainModel):
    """Machine-readable hierarchy index used for graph expansion in ask."""

    csc_id: str
    root: Path
    nodes: list[HierarchyIndexNode] = Field(default_factory=list)
    edges: list[HierarchyIndexEdge] = Field(default_factory=list)


class KnowledgeChunk(DomainModel):
    """Deterministic retrieval unit."""

    chunk_id: str
    source_type: Literal[
        "code",
        "sdd_section",
        "review_artifact",
        "file_summary",
        "directory_summary",
    ]
    source_path: Path
    text: str
    section_id: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    tokens: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalIndex(DomainModel):
    """Serialized lexical retrieval index."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    chunks: list[KnowledgeChunk]


class ChunkShardRef(DomainModel):
    """Metadata pointer for one chunk shard."""

    shard_id: int
    path: Path
    count: int


class PostingShardRef(DomainModel):
    """Metadata pointer for one postings shard."""

    shard_id: int
    bucket: str
    path: Path
    count: int


class DocStatRecord(DomainModel):
    """Per-chunk lexical statistics used by BM25 retrieval."""

    chunk_id: str
    doc_length: int


class RetrievalManifest(DomainModel):
    """Sharded retrieval index metadata."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: Literal["v1-sharded-json"] = "v1-sharded-json"
    shard_size: int
    total_chunks: int
    average_doc_length: float
    chunk_shards: list[ChunkShardRef] = Field(default_factory=list)
    posting_shards: list[PostingShardRef] = Field(default_factory=list)
    docstats_path: Path


class RunStageMetric(DomainModel):
    """Metrics for one workflow stage."""

    stage: str
    files_seen: int = 0
    chunks_written: int = 0
    chunks_loaded: int = 0
    elapsed_seconds: float = 0.0
    peak_rss_estimate: float = 0.0


class RunMetrics(DomainModel):
    """Run-level telemetry persisted for long-running workflows."""

    csc_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stages: list[RunStageMetric] = Field(default_factory=list)


class Citation(DomainModel):
    """Citation for grounded answer evidence."""

    chunk_id: str = Field(description="Referenced retrieval chunk identifier.")
    source_path: Path = Field(
        description="Relative source path for the cited evidence."
    )
    line_start: int | None = Field(
        default=None,
        description="Optional 1-based starting line number of cited span.",
    )
    line_end: int | None = Field(
        default=None,
        description="Optional 1-based ending line number of cited span.",
    )
    quote: str = Field(description="Short supporting excerpt from the cited chunk.")


class GraphInclusionPath(DomainModel):
    """Deterministic graph-path rationale for chunk inclusion."""

    distance: int = Field(
        ge=1,
        description="Traversal hop distance from an anchor node to this path target.",
    )
    edge_type: str = Field(description="Graph edge type used on this traversal step.")
    source_node_id: str = Field(description="Source node ID for the traversed edge.")
    source_node_type: str | None = Field(
        default=None,
        description="Optional source node type when available in the graph store.",
    )
    source_label: str | None = Field(
        default=None,
        description="Optional human-readable source node label for inspection.",
    )
    target_node_id: str = Field(description="Target node ID for the traversed edge.")
    target_node_type: str | None = Field(
        default=None,
        description="Optional target node type when available in the graph store.",
    )
    target_label: str | None = Field(
        default=None,
        description="Optional human-readable target node label for inspection.",
    )


class ChunkInclusionReason(DomainModel):
    """Score breakdown for why a chunk was selected for Q&A evidence."""

    chunk_id: str
    source: Literal["lexical", "hierarchy", "graph", "vector", "intensive"] = "lexical"
    lexical_score: float = 0.0
    anchor_score: float = 0.0
    graph_score: float = 0.0
    type_bias: float = 0.0
    final_score: float = 0.0
    reason: str
    graph_paths: list[GraphInclusionPath] = Field(
        default_factory=list,
        description=(
            "Optional graph traversal rationale that links this chunk to anchor "
            "evidence."
        ),
    )


class QueryRequest(DomainModel):
    """Question request passed to query workflow."""

    question: str
    top_k: int = 6
    session_history: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Trim user questions and prevent empty queries."""

        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class IntensiveCorpusSegment(DomainModel):
    """One file-bounded segment inside an intensive ask corpus chunk."""

    source_path: Path
    line_start: int
    line_end: int
    text: str
    token_count: int = Field(
        ge=0,
        description="Approximate token count for this segment using the project tokenizer.",
    )


class IntensiveCorpusChunk(DomainModel):
    """Large structured chunk screened during intensive ask mode."""

    chunk_id: str
    token_count: int = Field(
        ge=0,
        description="Approximate total token count across all ordered segments.",
    )
    segments: list[IntensiveCorpusSegment] = Field(default_factory=list)


class IntensiveCorpusManifest(DomainModel):
    """Manifest for the persisted intensive ask corpus store."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: Literal["v1-intensive-json"] = "v1-intensive-json"
    csc_id: str
    corpus_fingerprint: str
    chunk_tokens: int
    file_count: int
    chunk_count: int
    chunks_path: Path


class IntensiveScreeningExcerpt(DomainModel):
    """One excerpt selected by chunk-level intensive screening."""

    source_path: Path
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    reason: str


class IntensiveChunkScreening(DomainModel):
    """Structured relevance screening response for one corpus chunk."""

    chunk_id: str
    is_relevant: bool
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Model-estimated relevance of this chunk to the question.",
    )
    rationale: str
    selected_excerpts: list[IntensiveScreeningExcerpt] = Field(default_factory=list)


class IntensiveSelectedExcerpt(DomainModel):
    """Merged and ranked excerpt retained for the final intensive answer pass."""

    excerpt_id: str
    source_path: Path
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text: str
    reason: str
    screening_score: float = Field(ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)


class IntensiveRunManifest(DomainModel):
    """Manifest for one persisted intensive ask run."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: Literal["v1-intensive-run-json"] = "v1-intensive-run-json"
    question_hash: str
    question: str
    corpus_manifest_path: Path
    screenings_path: Path
    selected_excerpts_path: Path
    model_name: str
    temperature: float
    chunk_tokens: int
    max_selected_excerpts: int
    total_chunks_screened: int
    relevant_chunk_count: int
    selected_excerpt_count: int
    corpus_reused: bool = False


class QueryEvidencePack(DomainModel):
    """Evidence provided to the query prompt builder."""

    request: QueryRequest
    chunks: list[KnowledgeChunk]
    citations: list[Citation]
    primary_chunks: list[KnowledgeChunk] = Field(default_factory=list)
    related_files: list[Path] = Field(default_factory=list)
    related_symbols: list[str] = Field(default_factory=list)
    related_sections: list[str] = Field(default_factory=list)
    related_commits: list[str] = Field(default_factory=list)
    inclusion_reasons: list[ChunkInclusionReason] = Field(default_factory=list)


class QueryAnswer(DomainModel):
    """Structured query answer result."""

    answer: str = Field(
        description=("Grounded answer text using only supplied evidence and citations.")
    )
    citations: list[Citation] = Field(
        description=("Citations supporting factual claims in the answer.")
    )
    uncertainty: list[str] = Field(
        default_factory=list,
        description=("Known caveats, confidence limits, or fallback conditions."),
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Information needed for a complete answer but absent from provided "
            "evidence. Use TBD items when unknown."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence that the answer is supported by cited evidence "
            "(0.0 low, 1.0 high)."
        ),
    )


class ScanResult(DomainModel):
    """Repository scan output for downstream workflows."""

    files: list[Path]
    code_summaries: list[CodeUnitSummary]
    symbol_summaries: list[SymbolSummary]
    dependencies: list[str]
    code_chunks: list[KnowledgeChunk]


class GenerateResult(DomainModel):
    """Output of generate workflow."""

    document: SDDDocument
    review_artifact: ReviewArtifact
    retrieval_manifest: RetrievalManifest
    markdown_path: Path
    review_json_path: Path
    retrieval_index_path: Path
    run_metrics_path: Path
    hierarchy_manifest_path: Path | None = None
    hierarchy_store_path: Path | None = None
    graph_manifest_path: Path | None = None
    graph_store_path: Path | None = None


class ProposeUpdatesResult(DomainModel):
    """Output of propose-updates workflow."""

    impact: CommitImpact
    report: UpdateProposalReport
    retrieval_manifest: RetrievalManifest
    report_markdown_path: Path
    report_json_path: Path
    retrieval_index_path: Path
    run_metrics_path: Path
    hierarchy_manifest_path: Path | None = None
    hierarchy_store_path: Path | None = None
    graph_manifest_path: Path | None = None
    graph_store_path: Path | None = None


class InspectDiffResult(DomainModel):
    """Output model for inspect-diff command."""

    impact: CommitImpact
    raw_diff: str


class AskResult(DomainModel):
    """Output of ask workflow."""

    answer: QueryAnswer
    evidence_pack: QueryEvidencePack
