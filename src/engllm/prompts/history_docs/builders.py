"""Deterministic prompt builders for the history-docs tool."""

from __future__ import annotations

import json

from engllm.prompts.history_docs.templates import (
    DEPENDENCY_SUMMARY_SYSTEM_PROMPT,
    HISTORY_DOCS_QUALITY_JUDGE_SYSTEM_PROMPT,
)
from engllm.tools.history_docs.models import (
    HistoryCheckpointModel,
    HistoryDependencyEntry,
    HistoryDocsBenchmarkCase,
    HistoryRenderManifest,
    HistoryValidationReport,
)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_dependency_summary_prompt(
    entry: HistoryDependencyEntry,
) -> tuple[str, str]:
    """Return prompts for one dependency summary generation request."""

    user_prompt = (
        "Document this direct dependency using only the provided evidence.\n"
        f"Dependency Evidence:\n{_json(entry.model_dump(mode='json'))}\n"
    )
    return DEPENDENCY_SUMMARY_SYSTEM_PROMPT, user_prompt


def build_history_docs_quality_judge_prompt(
    *,
    case: HistoryDocsBenchmarkCase,
    markdown: str,
    render_manifest: HistoryRenderManifest,
    validation_report: HistoryValidationReport,
    checkpoint_model: HistoryCheckpointModel,
) -> tuple[str, str]:
    """Return prompts for one H10 single-document quality evaluation."""

    checkpoint_summary = {
        "checkpoint_id": checkpoint_model.checkpoint_id,
        "subsystem_count": len(checkpoint_model.subsystems),
        "module_count": len(checkpoint_model.modules),
        "dependency_concept_count": len(checkpoint_model.dependencies),
        "algorithm_capsule_count": len(checkpoint_model.algorithm_capsule_ids),
        "rendered_section_count": len(render_manifest.sections),
        "validation_error_count": validation_report.error_count,
        "validation_warning_count": validation_report.warning_count,
    }
    render_sections = [
        {
            "section_id": section.section_id,
            "title": section.title,
            "kind": section.kind,
            "dependency_ids": section.dependency_ids,
            "algorithm_capsule_ids": section.algorithm_capsule_ids,
        }
        for section in render_manifest.sections
    ]
    validation_summary = {
        "error_count": validation_report.error_count,
        "warning_count": validation_report.warning_count,
        "findings": [
            {
                "check_id": finding.check_id,
                "severity": finding.severity,
                "section_id": finding.section_id,
                "reference": finding.reference,
            }
            for finding in validation_report.findings
        ],
    }
    user_prompt = (
        "Evaluate this rendered history-docs checkpoint document against the "
        "benchmark expectations.\n"
        f"Benchmark Case:\n{_json(case.model_dump(mode='json'))}\n"
        f"Checkpoint Summary:\n{_json(checkpoint_summary)}\n"
        f"Rendered Sections:\n{_json(render_sections)}\n"
        f"Validation Summary:\n{_json(validation_summary)}\n"
        "Checkpoint Markdown:\n"
        f"{markdown}\n"
    )
    return HISTORY_DOCS_QUALITY_JUDGE_SYSTEM_PROMPT, user_prompt
