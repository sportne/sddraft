"""Typed domain models for the SDDraft pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainModel(BaseModel):
    """Base model with strict shape checking."""

    model_config = ConfigDict(extra="forbid")


SourceLanguage = Literal["python", "java", "cpp", "unknown"]


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
        if value < 0.0 or value > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return value


class GenerationOptions(DomainModel):
    """Optional generation tuning knobs."""

    max_files: int = 500
    code_chunk_lines: int = 40
    retrieval_top_k: int = 6


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


class InterfaceSummary(DomainModel):
    """Public interface summary for one type/function source."""

    name: str
    kind: Literal["class", "function", "module"]
    language: SourceLanguage
    source_path: Path
    members: list[str] = Field(default_factory=list)


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

    kind: str
    source: str
    detail: str | None = None


class SectionEvidencePack(DomainModel):
    """Deterministic section-scoped evidence payload."""

    section: SDDSectionSpec
    csc: CSCDescriptor
    code_summaries: list[CodeUnitSummary] = Field(default_factory=list)
    interface_summaries: list[InterfaceSummary] = Field(default_factory=list)
    dependency_summaries: list[str] = Field(default_factory=list)
    commit_impact: CommitImpact | None = None
    existing_section_text: str | None = None
    evidence_references: list[EvidenceReference] = Field(default_factory=list)


class SectionDraft(DomainModel):
    """Structured generated section draft."""

    section_id: str
    title: str
    content: str
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class SectionUpdateProposal(DomainModel):
    """Structured section update proposal."""

    section_id: str
    title: str
    existing_text: str
    proposed_text: str
    rationale: str
    uncertainty_list: list[str] = Field(default_factory=list)
    review_priority: Literal["low", "medium", "high"] = "medium"
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)


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


class KnowledgeChunk(DomainModel):
    """Deterministic retrieval unit."""

    chunk_id: str
    source_type: Literal["code", "sdd_section", "review_artifact"]
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


class Citation(DomainModel):
    """Citation for grounded answer evidence."""

    chunk_id: str
    source_path: Path
    line_start: int | None = None
    line_end: int | None = None
    quote: str


class QueryRequest(DomainModel):
    """Question request passed to query workflow."""

    question: str
    top_k: int = 6
    session_history: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class QueryEvidencePack(DomainModel):
    """Evidence provided to the query prompt builder."""

    request: QueryRequest
    chunks: list[KnowledgeChunk]
    citations: list[Citation]


class QueryAnswer(DomainModel):
    """Structured query answer result."""

    answer: str
    citations: list[Citation]
    uncertainty: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class ScanResult(DomainModel):
    """Repository scan output for downstream workflows."""

    files: list[Path]
    code_summaries: list[CodeUnitSummary]
    interface_summaries: list[InterfaceSummary]
    dependencies: list[str]
    code_chunks: list[KnowledgeChunk]


class GenerateResult(DomainModel):
    """Output of generate workflow."""

    document: SDDDocument
    review_artifact: ReviewArtifact
    retrieval_index: RetrievalIndex
    markdown_path: Path
    review_json_path: Path
    retrieval_index_path: Path


class ProposeUpdatesResult(DomainModel):
    """Output of propose-updates workflow."""

    impact: CommitImpact
    report: UpdateProposalReport
    retrieval_index: RetrievalIndex
    report_markdown_path: Path
    report_json_path: Path
    retrieval_index_path: Path


class InspectDiffResult(DomainModel):
    """Output model for inspect-diff command."""

    impact: CommitImpact
    raw_diff: str


class AskResult(DomainModel):
    """Output of ask workflow."""

    answer: QueryAnswer
    evidence_pack: QueryEvidencePack
