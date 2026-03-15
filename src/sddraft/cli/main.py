"""SDDraft command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sddraft.config.loader import load_config_bundle
from sddraft.domain.errors import SDDraftError
from sddraft.domain.models import QueryRequest
from sddraft.llm.factory import create_llm_client
from sddraft.render.reports import render_query_answer_text
from sddraft.workflows.ask import answer_question
from sddraft.workflows.generate import generate_sdd
from sddraft.workflows.inspect_diff import inspect_diff
from sddraft.workflows.propose_updates import propose_updates


def _progress(message: str) -> None:
    print(f"[progress] {message}")


def _add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-config", required=True, type=Path)
    parser.add_argument("--csc", required=True, nargs="+", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--provider", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-hierarchy-docs", action="store_true")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sddraft")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config_parser = subparsers.add_parser("validate-config")
    validate_config_parser.add_argument("--project-config", required=True, type=Path)
    validate_config_parser.add_argument("--csc", required=True, nargs="+", type=Path)
    validate_config_parser.add_argument("--template", type=Path)

    generate_parser = subparsers.add_parser("generate")
    _add_common_generation_args(generate_parser)

    propose_parser = subparsers.add_parser("propose-updates")
    _add_common_generation_args(propose_parser)
    propose_parser.add_argument("--existing-sdd", required=True, type=Path)
    propose_parser.add_argument("--commit-range", required=True)

    inspect_parser = subparsers.add_parser("inspect-diff")
    inspect_parser.add_argument("--commit-range", required=True)
    inspect_parser.add_argument("--repo-root", type=Path, default=Path("."))

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("--index-path", required=True, type=Path)
    ask_parser.add_argument("--question", type=str)
    ask_parser.add_argument("--interactive", action="store_true")
    ask_parser.add_argument("--provider", type=str, default="mock")
    ask_parser.add_argument("--model", type=str, default="mock-sddraft")
    ask_parser.add_argument("--temperature", type=float, default=0.2)
    ask_parser.add_argument("--top-k", type=int, default=6)

    return parser.parse_args(argv)


def _run_validate_config(args: argparse.Namespace) -> int:
    bundle = load_config_bundle(
        project_config_path=args.project_config,
        csc_paths=list(args.csc),
        template_path=args.template,
    )
    print(
        f"Configuration is valid for project '{bundle.project.project_name}' "
        f"with {len(bundle.csc_descriptors)} CSC descriptor(s)."
    )
    return 0


def _resolve_temperature(raw_value: float | None, default: float) -> float:
    resolved = default if raw_value is None else raw_value
    if resolved < 0.0 or resolved > 1.0:
        raise SDDraftError("--temperature must be between 0.0 and 1.0")
    return resolved


def _run_generate(args: argparse.Namespace) -> int:
    bundle = load_config_bundle(
        project_config_path=args.project_config,
        csc_paths=list(args.csc),
        template_path=args.template,
    )
    resolved_model = args.model or bundle.project.llm.model_name
    resolved_temperature = _resolve_temperature(
        args.temperature, bundle.project.llm.temperature
    )

    llm_client = create_llm_client(
        bundle.project.llm, provider=args.provider, model_name=resolved_model
    )

    for csc in bundle.csc_descriptors:
        result = generate_sdd(
            project_config=bundle.project,
            csc=csc,
            template=bundle.template,
            llm_client=llm_client,
            repo_root=args.repo_root.resolve(),
            model_name=resolved_model,
            temperature=resolved_temperature,
            hierarchy_docs_enabled=not args.no_hierarchy_docs,
            progress_callback=_progress,
        )
        print(
            f"Generated SDD for {csc.csc_id}: "
            f"{result.markdown_path} "
            f"(review: {result.review_json_path}, index: {result.retrieval_index_path})"
        )
    return 0


def _run_propose_updates(args: argparse.Namespace) -> int:
    bundle = load_config_bundle(
        project_config_path=args.project_config,
        csc_paths=list(args.csc),
        template_path=args.template,
    )
    resolved_model = args.model or bundle.project.llm.model_name
    resolved_temperature = _resolve_temperature(
        args.temperature, bundle.project.llm.temperature
    )

    llm_client = create_llm_client(
        bundle.project.llm, provider=args.provider, model_name=resolved_model
    )

    for csc in bundle.csc_descriptors:
        result = propose_updates(
            project_config=bundle.project,
            csc=csc,
            template=bundle.template,
            llm_client=llm_client,
            existing_sdd_path=args.existing_sdd,
            commit_range=args.commit_range,
            repo_root=args.repo_root.resolve(),
            model_name=resolved_model,
            temperature=resolved_temperature,
            hierarchy_docs_enabled=not args.no_hierarchy_docs,
            progress_callback=_progress,
        )
        print(
            f"Generated update proposals for {csc.csc_id}: "
            f"{result.report_markdown_path} "
            f"(json: {result.report_json_path}, index: {result.retrieval_index_path})"
        )
    return 0


def _run_inspect_diff(args: argparse.Namespace) -> int:
    result = inspect_diff(
        commit_range=args.commit_range, repo_root=args.repo_root.resolve()
    )
    print(result.impact.model_dump_json(indent=2))
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    from sddraft.domain.models import LLMConfig

    resolved_temperature = _resolve_temperature(args.temperature, 0.2)
    llm_client = create_llm_client(
        LLMConfig(
            provider=args.provider,
            model_name=args.model,
            temperature=resolved_temperature,
        )
    )

    if args.interactive:
        history: list[str] = []
        while True:
            question = input("sddraft ask> ").strip()
            if not question or question.lower() in {"exit", "quit"}:
                break
            request = QueryRequest(
                question=question, top_k=args.top_k, session_history=history
            )
            result = answer_question(
                request=request,
                index_path=args.index_path,
                llm_client=llm_client,
                model_name=args.model,
                temperature=resolved_temperature,
            )
            print(render_query_answer_text(result.answer))
            history.extend([f"Q: {question}", f"A: {result.answer.answer}"])
        return 0

    if not args.question:
        raise SDDraftError("--question is required unless --interactive is provided")

    request = QueryRequest(question=args.question, top_k=args.top_k)
    result = answer_question(
        request=request,
        index_path=args.index_path,
        llm_client=llm_client,
        model_name=args.model,
        temperature=resolved_temperature,
    )
    print(render_query_answer_text(result.answer))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    args = _parse_args(argv)
    try:
        if args.command == "validate-config":
            return _run_validate_config(args)
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "propose-updates":
            return _run_propose_updates(args)
        if args.command == "inspect-diff":
            return _run_inspect_diff(args)
        if args.command == "ask":
            return _run_ask(args)
    except SDDraftError as exc:
        print(f"Error: {exc}")
        return 2

    print(json.dumps({"error": f"Unknown command {args.command}"}))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
