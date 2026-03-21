"""H11 semantic checkpoint-planning helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from engllm.core.analysis.commit_impact import build_commit_impact
from engllm.core.analysis.history import HistoryCheckpoint, HistoryCommitSummary
from engllm.core.repo.diff_parser import (
    extract_changed_symbol_names,
    get_git_diff_between,
    parse_diff,
)
from engllm.core.repo.history import (
    describe_commit_diff,
    get_commit_parents,
    iter_first_parent_commits,
    list_reachable_tags_by_commit,
    list_tree_paths_at_commit,
)
from engllm.domain.errors import RepositoryError
from engllm.llm.base import LLMClient, StructuredGenerationRequest
from engllm.prompts.history_docs import build_semantic_checkpoint_planner_prompt
from engllm.tools.history_docs.models import (
    HistoryEvidenceLink,
    HistorySemanticCheckpointCandidate,
    HistorySemanticCheckpointEvaluationStatus,
    HistorySemanticCheckpointJudgmentBatch,
    HistorySemanticCheckpointPlan,
    HistorySemanticCheckpointRecommendation,
    HistorySemanticCheckpointSignalKind,
)
from engllm.tools.history_docs.structure import normalize_relative_path

_MAX_LLM_CANDIDATES = 12
_BUILD_SOURCE_NAMES = {
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.toml",
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "settings.gradle.kts",
    "gradle.properties",
    "packages.lock.json",
    "Directory.Build.props",
    "global.json",
    "CMakeLists.txt",
    "vcpkg.json",
    "conanfile.py",
    "conanfile.txt",
    "meson.build",
    "BUILD",
    "WORKSPACE",
    "Makefile",
}
_RECOMMENDATION_ORDER: dict[HistorySemanticCheckpointRecommendation, int] = {
    "primary": 0,
    "supporting": 1,
    "skip": 2,
}


def semantic_checkpoint_plan_path(tool_root: Path, checkpoint_id: str) -> Path:
    """Return the H11 semantic checkpoint plan artifact path."""

    return tool_root / "checkpoints" / checkpoint_id / "semantic_checkpoint_plan.json"


def _timestamp_key(timestamp: str) -> float:
    return datetime.fromisoformat(timestamp).timestamp()


def _sort_evidence_links(
    links: list[HistoryEvidenceLink],
) -> list[HistoryEvidenceLink]:
    unique = {(link.kind, link.reference, link.detail or ""): link for link in links}
    return [unique[key] for key in sorted(unique)]


def _sort_candidates_for_prompt(
    candidates: list[HistorySemanticCheckpointCandidate],
) -> list[HistorySemanticCheckpointCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.heuristic_score,
            -_timestamp_key(item.commit.timestamp),
            item.commit.sha,
        ),
    )


def _sort_candidates_for_output(
    candidates: list[HistorySemanticCheckpointCandidate],
) -> list[HistorySemanticCheckpointCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            _RECOMMENDATION_ORDER[item.recommendation],
            -item.heuristic_score,
            -_timestamp_key(item.commit.timestamp),
            item.commit.sha,
        ),
    )


def _resolve_source_roots(repo_root: Path, configured_roots: list[Path]) -> list[Path]:
    resolved_roots: list[Path] = []
    for configured_root in configured_roots:
        absolute_root = (
            configured_root
            if configured_root.is_absolute()
            else (repo_root / configured_root)
        ).resolve()
        try:
            relative_root = normalize_relative_path(
                absolute_root.relative_to(repo_root.resolve())
            )
        except ValueError as exc:
            raise RepositoryError(
                "Configured source root must live inside repo root for history-docs: "
                f"{configured_root}"
            ) from exc
        resolved_roots.append(relative_root)
    return resolved_roots


def _is_build_source_path(path: Path) -> bool:
    name = path.name
    return (
        name in _BUILD_SOURCE_NAMES
        or (name.startswith("requirements-") and name.endswith(".txt"))
        or path.suffix == ".csproj"
    )


def _top_level_area_label(source_root: Path, area_name: str) -> str:
    if source_root == Path("."):
        return area_name
    return f"{source_root.as_posix()}/{area_name}"


def _top_level_areas_for_paths(
    changed_paths: list[Path],
    source_roots: list[Path],
) -> dict[str, Path]:
    labels_to_prefixes: dict[str, Path] = {}
    for changed_path in changed_paths:
        normalized_path = normalize_relative_path(changed_path)
        for source_root in source_roots:
            if source_root == Path("."):
                relative_path = normalized_path
            else:
                try:
                    relative_path = normalized_path.relative_to(source_root)
                except ValueError:
                    continue
            if not relative_path.parts:
                continue
            label = _top_level_area_label(source_root, relative_path.parts[0])
            prefix = (
                Path(relative_path.parts[0])
                if source_root == Path(".")
                else source_root / relative_path.parts[0]
            )
            labels_to_prefixes.setdefault(label, prefix)
    return labels_to_prefixes


def _introduced_top_level_areas(
    *,
    repo_root: Path,
    base_rev: str,
    target_rev: str,
    top_level_area_prefixes: dict[str, Path],
) -> list[str]:
    introduced: list[str] = []
    for label, prefix in top_level_area_prefixes.items():
        if list_tree_paths_at_commit(
            repo_root,
            target_rev,
            prefix=prefix,
        ) and not list_tree_paths_at_commit(repo_root, base_rev, prefix=prefix):
            introduced.append(label)
    return sorted(set(introduced))


def _heuristic_score(
    signal_kinds: list[HistorySemanticCheckpointSignalKind],
) -> int:
    signal_set = set(signal_kinds)
    score = 0
    if "tag_anchor" in signal_set:
        score += 3
    if "interface_shift" in signal_set:
        score += 2
    if "dependency_shift" in signal_set or "build_shift" in signal_set:
        score += 2
    if "broad_change" in signal_set:
        score += 1
    if "new_top_level_area" in signal_set:
        score += 1
    if "merge_anchor" in signal_set:
        score += 1
    return score


def _heuristic_recommendation(
    score: int,
) -> HistorySemanticCheckpointRecommendation:
    return "primary" if score >= 5 else "supporting"


def _heuristic_title(candidate: HistorySemanticCheckpointCandidate) -> str:
    signal_set = set(candidate.signal_kinds)
    if candidate.tag_names:
        return f"Tagged milestone: {candidate.tag_names[0]}"
    if "new_top_level_area" in signal_set and candidate.top_level_areas:
        return f"New area: {candidate.top_level_areas[0]}"
    if "interface_shift" in signal_set:
        return "Interface boundary shift"
    if "build_shift" in signal_set or "dependency_shift" in signal_set:
        return "Dependency or build milestone"
    if "broad_change" in signal_set:
        return "Broad structural change"
    if "merge_anchor" in signal_set:
        return "Merge consolidation point"
    return candidate.commit.subject


def _heuristic_rationale(candidate: HistorySemanticCheckpointCandidate) -> str:
    signal_phrase = ", ".join(candidate.signal_kinds) or "no strong signals"
    area_phrase = (
        f" across {', '.join(candidate.top_level_areas)}"
        if candidate.top_level_areas
        else ""
    )
    return (
        f"Commit '{candidate.commit.subject}' is retained as a semantic checkpoint "
        f"candidate because it carries {signal_phrase}{area_phrase}."
    )


def _default_uncertainty(
    status: HistorySemanticCheckpointEvaluationStatus,
) -> str | None:
    if status == "heuristic_only":
        return (
            "No semantic planner judgments were returned; recommendations are based "
            "on deterministic history signals only."
        )
    if status == "llm_failed":
        return (
            "The semantic planner fell back to deterministic heuristics after the "
            "LLM evaluation step failed."
        )
    return None


def _planner_prompt_candidates(
    candidates: list[HistorySemanticCheckpointCandidate],
) -> list[dict[str, object]]:
    return [
        {
            "candidate_commit_id": candidate.commit.sha,
            "short_sha": candidate.commit.short_sha,
            "timestamp": candidate.commit.timestamp,
            "subject": candidate.commit.subject,
            "window_start_commit": candidate.window_start_commit,
            "window_commit_count": candidate.window_commit_count,
            "tag_names": candidate.tag_names,
            "top_level_areas": candidate.top_level_areas,
            "change_kinds": candidate.change_kinds,
            "signal_kinds": candidate.signal_kinds,
            "heuristic_score": candidate.heuristic_score,
        }
        for candidate in candidates
    ]


def _built_checkpoint_payload(
    checkpoints: list[HistoryCheckpoint],
) -> list[dict[str, object]]:
    return [
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "target_commit": checkpoint.target_commit,
            "target_commit_timestamp": checkpoint.target_commit_timestamp,
            "target_commit_subject": checkpoint.target_commit_subject,
        }
        for checkpoint in checkpoints
    ]


def _merge_judgments(
    *,
    candidates: list[HistorySemanticCheckpointCandidate],
    judgments: HistorySemanticCheckpointJudgmentBatch,
    evaluation_status: HistorySemanticCheckpointEvaluationStatus,
) -> list[HistorySemanticCheckpointCandidate]:
    judgments_by_commit = {
        judgment.candidate_commit_id: judgment for judgment in judgments.judgments
    }
    merged: list[HistorySemanticCheckpointCandidate] = []
    for candidate in candidates:
        default_recommendation = _heuristic_recommendation(candidate.heuristic_score)
        semantic_title = _heuristic_title(candidate)
        rationale = _heuristic_rationale(candidate)
        uncertainty = _default_uncertainty(evaluation_status)
        recommendation = default_recommendation

        judgment = judgments_by_commit.get(candidate.commit.sha)
        if judgment is not None:
            recommendation = judgment.recommendation
            semantic_title = judgment.semantic_title.strip() or semantic_title
            rationale = judgment.rationale.strip() or rationale
            uncertainty = (
                judgment.uncertainty.strip()
                if judgment.uncertainty is not None and judgment.uncertainty.strip()
                else None
            )

        merged.append(
            candidate.model_copy(
                update={
                    "recommendation": recommendation,
                    "semantic_title": semantic_title,
                    "rationale": rationale,
                    "uncertainty": uncertainty,
                }
            )
        )
    return merged


def _candidate_window_annotations(
    candidates: list[HistorySemanticCheckpointCandidate],
    ancestry: list[HistoryCommitSummary],
) -> dict[str, tuple[str | None, int]]:
    index_by_commit = {commit.sha: index for index, commit in enumerate(ancestry)}
    chronological = sorted(
        candidates, key=lambda item: index_by_commit[item.commit.sha]
    )
    windows: dict[str, tuple[str | None, int]] = {}
    previous_index = -1
    for candidate in chronological:
        candidate_index = index_by_commit[candidate.commit.sha]
        start_index = previous_index + 1
        window_start_commit = (
            ancestry[start_index].sha if 0 <= start_index < len(ancestry) else None
        )
        window_commit_count = candidate_index - previous_index
        windows[candidate.commit.sha] = (window_start_commit, window_commit_count)
        previous_index = candidate_index
    return windows


def _build_candidate(
    *,
    commit: HistoryCommitSummary,
    source_roots: list[Path],
    repo_root: Path,
    tag_names: list[str],
) -> HistorySemanticCheckpointCandidate | None:
    diff_spec = describe_commit_diff(repo_root, commit.sha)
    diff_text = get_git_diff_between(diff_spec.base_rev, commit.sha, repo_root)
    file_diffs = parse_diff(diff_text)
    impact = build_commit_impact(f"{diff_spec.base_rev}..{commit.sha}", file_diffs)
    changed_paths = sorted(item.path for item in impact.changed_files)
    build_source_paths = [path for path in changed_paths if _is_build_source_path(path)]
    top_level_map = _top_level_areas_for_paths(changed_paths, source_roots)
    top_level_areas = sorted(top_level_map)
    introduced_top_level_areas = _introduced_top_level_areas(
        repo_root=repo_root,
        base_rev=diff_spec.base_rev,
        target_rev=commit.sha,
        top_level_area_prefixes=top_level_map,
    )
    parent_count = len(get_commit_parents(repo_root, commit.sha))

    signal_kinds: list[HistorySemanticCheckpointSignalKind] = []
    if tag_names:
        signal_kinds.append("tag_anchor")
    if "interface_change" in impact.change_kinds:
        signal_kinds.append("interface_shift")
    if "dependency_change" in impact.change_kinds:
        signal_kinds.append("dependency_shift")
    if build_source_paths:
        signal_kinds.append("build_shift")
    if len(changed_paths) >= 4 or len(top_level_areas) >= 2:
        signal_kinds.append("broad_change")
    if introduced_top_level_areas:
        signal_kinds.append("new_top_level_area")
    if parent_count > 1:
        signal_kinds.append("merge_anchor")

    if not signal_kinds:
        return None

    changed_symbols = sorted(
        {
            symbol
            for diff in impact.changed_files
            for symbol in extract_changed_symbol_names(diff.signature_changes)
        }
    )
    evidence_links = [HistoryEvidenceLink(kind="commit", reference=commit.sha)]
    evidence_links.extend(
        HistoryEvidenceLink(kind="build_source", reference=path.as_posix())
        for path in build_source_paths
    )
    evidence_links.extend(
        HistoryEvidenceLink(kind="file", reference=path.as_posix())
        for path in changed_paths[:6]
    )
    evidence_links.extend(
        HistoryEvidenceLink(kind="symbol", reference=symbol)
        for symbol in changed_symbols[:6]
    )
    for tag_name in tag_names:
        evidence_links.append(
            HistoryEvidenceLink(
                kind="commit", reference=commit.sha, detail=f"tag:{tag_name}"
            )
        )

    heuristic_score = _heuristic_score(signal_kinds)
    return HistorySemanticCheckpointCandidate(
        commit=commit,
        tag_names=sorted(tag_names),
        top_level_areas=top_level_areas,
        change_kinds=sorted(impact.change_kinds),
        signal_kinds=sorted(signal_kinds),
        heuristic_score=heuristic_score,
        recommendation=_heuristic_recommendation(heuristic_score),
        semantic_title=commit.subject,
        rationale="",
        evidence_links=_sort_evidence_links(evidence_links),
    )


def build_semantic_checkpoint_plan(
    *,
    repo_root: Path,
    checkpoint_id: str,
    target_commit: str,
    previous_checkpoint_commit: str | None,
    configured_source_roots: list[Path],
    checkpoints: list[HistoryCheckpoint],
    llm_client: LLMClient,
    model_name: str,
    temperature: float,
) -> HistorySemanticCheckpointPlan:
    """Build the H11 semantic checkpoint plan for one requested checkpoint."""

    source_roots = _resolve_source_roots(repo_root, configured_source_roots)
    ancestry = [
        HistoryCommitSummary(
            sha=commit.sha,
            short_sha=commit.short_sha,
            timestamp=commit.timestamp,
            subject=commit.subject,
        )
        for commit in iter_first_parent_commits(
            repo_root,
            target_commit=target_commit,
        )
    ]
    commit_shas = [commit.sha for commit in ancestry]
    tags_by_commit = list_reachable_tags_by_commit(
        repo_root,
        target_commit=target_commit,
        commit_shas=commit_shas,
    )

    candidates: list[HistorySemanticCheckpointCandidate] = []
    for commit in ancestry:
        candidate = _build_candidate(
            commit=commit,
            source_roots=source_roots,
            repo_root=repo_root,
            tag_names=tags_by_commit.get(commit.sha, []),
        )
        if candidate is not None:
            candidates.append(candidate)

    prompt_candidates = _sort_candidates_for_prompt(candidates)[:_MAX_LLM_CANDIDATES]
    target_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.commit.sha == target_commit
        ),
        None,
    )
    if target_candidate is not None and all(
        candidate.commit.sha != target_commit for candidate in prompt_candidates
    ):
        if len(prompt_candidates) < _MAX_LLM_CANDIDATES:
            prompt_candidates = [*prompt_candidates, target_candidate]
        elif prompt_candidates:
            prompt_candidates = [*prompt_candidates[:-1], target_candidate]
        prompt_candidates = _sort_candidates_for_prompt(prompt_candidates)
    windows = _candidate_window_annotations(prompt_candidates, ancestry)
    annotated_prompt_candidates = [
        candidate.model_copy(
            update={
                "window_start_commit": windows[candidate.commit.sha][0],
                "window_commit_count": windows[candidate.commit.sha][1],
                "rationale": _heuristic_rationale(candidate),
            }
        )
        for candidate in prompt_candidates
    ]

    evaluation_status: HistorySemanticCheckpointEvaluationStatus = "heuristic_only"
    merged_candidates = annotated_prompt_candidates
    if annotated_prompt_candidates:
        try:
            system_prompt, user_prompt = build_semantic_checkpoint_planner_prompt(
                checkpoint_id=checkpoint_id,
                target_commit=target_commit,
                previous_checkpoint_commit=previous_checkpoint_commit,
                built_checkpoints=_built_checkpoint_payload(checkpoints),
                candidates=_planner_prompt_candidates(annotated_prompt_candidates),
            )
            response = llm_client.generate_structured(
                StructuredGenerationRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=HistorySemanticCheckpointJudgmentBatch,
                    model_name=model_name,
                    temperature=temperature,
                )
            )
            judgments = HistorySemanticCheckpointJudgmentBatch.model_validate(
                response.content.model_dump(mode="python")
            )
            filtered_judgments = HistorySemanticCheckpointJudgmentBatch(
                judgments=[
                    judgment
                    for judgment in judgments.judgments
                    if judgment.candidate_commit_id
                    in {
                        candidate.commit.sha
                        for candidate in annotated_prompt_candidates
                    }
                ]
            )
            if filtered_judgments.judgments:
                evaluation_status = "scored"
            merged_candidates = _merge_judgments(
                candidates=annotated_prompt_candidates,
                judgments=filtered_judgments,
                evaluation_status=evaluation_status,
            )
        except Exception:
            evaluation_status = "llm_failed"
            merged_candidates = [
                candidate.model_copy(
                    update={
                        "recommendation": "supporting",
                        "semantic_title": "TBD",
                        "rationale": _heuristic_rationale(candidate),
                        "uncertainty": _default_uncertainty("llm_failed"),
                    }
                )
                for candidate in annotated_prompt_candidates
            ]
    merged_candidates = [
        candidate.model_copy(
            update={
                "uncertainty": candidate.uncertainty
                or _default_uncertainty(evaluation_status),
                "rationale": candidate.rationale or _heuristic_rationale(candidate),
            }
        )
        for candidate in merged_candidates
    ]
    merged_candidates = _sort_candidates_for_output(merged_candidates)

    return HistorySemanticCheckpointPlan(
        checkpoint_id=checkpoint_id,
        target_commit=target_commit,
        previous_checkpoint_commit=previous_checkpoint_commit,
        evaluation_status=evaluation_status,
        current_target_recommended=any(
            candidate.commit.sha == target_commit and candidate.recommendation != "skip"
            for candidate in merged_candidates
        ),
        candidates=merged_candidates,
    )
