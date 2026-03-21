"""EngLLM command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from engllm.core.analysis.retrieval import migrate_legacy_index
from engllm.core.config.loader import load_config_bundle, load_project_config
from engllm.core.tooling import ToolCommand, ToolSpec
from engllm.domain.errors import EngLLMError
from engllm.domain.models import LLMConfig
from engllm.llm.base import LLMClient
from engllm.llm.factory import create_llm_client
from engllm.tools.ask.ask import answer_question
from engllm.tools.ask.models import AskMode, QueryRequest
from engllm.tools.ask.render import render_query_answer_text
from engllm.tools.repo.inspect_diff import inspect_diff
from engllm.tools.sdd.generate import generate_sdd
from engllm.tools.sdd.models import ProjectConfig
from engllm.tools.sdd.propose_updates import propose_updates


def _progress(message: str) -> None:
    print(f"[progress] {message}")


def _resolve_temperature(raw_value: float | None, default: float) -> float:
    resolved = default if raw_value is None else raw_value
    if resolved < 0.0 or resolved > 1.0:
        raise EngLLMError("--temperature must be between 0.0 and 1.0")
    return resolved


def _add_sdd_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True, nargs="+", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--provider", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--no-hierarchy-docs", action="store_true")
    parser.add_argument("--no-graph", action="store_true")


def _add_ask_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--provider", type=str, default="mock")
    parser.add_argument("--model", type=str, default="mock-engllm")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mode", choices=["standard", "intensive"])
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--graph-depth", type=int, choices=[1, 2], default=1)
    parser.add_argument("--graph-top-k", type=int, default=12)
    parser.add_argument(
        "--vector-enabled",
        dest="vector_enabled",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--no-vector-enabled",
        dest="vector_enabled",
        action="store_const",
        const=False,
        default=None,
    )
    parser.add_argument("--vector-top-k", type=int)
    parser.add_argument("--intensive-chunk-tokens", type=int)
    parser.add_argument("--intensive-max-selected-excerpts", type=int)


def _add_sdd_propose_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for commit-driven SDD proposal generation."""

    _add_sdd_common_args(parser)
    parser.add_argument("--existing-sdd", required=True, type=Path)
    parser.add_argument("--commit-range", required=True)


def _add_ask_answer_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for one-shot ask answers."""

    _add_ask_common_args(parser)
    parser.add_argument("--question", required=True)


def _add_repo_inspect_diff_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for diff inspection."""

    parser.add_argument("--commit-range", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))


def _add_repo_migrate_index_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for retrieval index migration."""

    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--write-batch-size", type=int, default=200)
    parser.add_argument("--max-in-memory-records", type=int, default=2000)


def _run_sdd_validate_config(args: argparse.Namespace) -> int:
    bundle = load_config_bundle(
        project_config_path=args.config,
        csc_paths=list(args.target),
        template_path=args.template,
    )
    print(
        f"Configuration is valid for project '{bundle.project.project_name}' "
        f"with {len(bundle.csc_descriptors)} target(s)."
    )
    return 0


def _run_sdd_generate(args: argparse.Namespace) -> int:
    bundle = load_config_bundle(
        project_config_path=args.config,
        csc_paths=list(args.target),
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
            graph_enabled=not args.no_graph,
            progress_callback=_progress,
        )
        print(
            f"Generated SDD for {csc.csc_id}: "
            f"{result.markdown_path} "
            f"(review: {result.review_json_path}, shared retrieval: {result.retrieval_index_path})"
        )
    return 0


def _run_sdd_propose_updates(args: argparse.Namespace) -> int:
    bundle = load_config_bundle(
        project_config_path=args.config,
        csc_paths=list(args.target),
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
            graph_enabled=not args.no_graph,
            progress_callback=_progress,
        )
        print(
            f"Generated update proposals for {csc.csc_id}: "
            f"{result.report_markdown_path} "
            f"(json: {result.report_json_path}, shared retrieval: {result.retrieval_index_path})"
        )
    return 0


def _resolve_ask_settings(
    args: argparse.Namespace,
) -> tuple[ProjectConfig | None, AskMode, bool, int, int, int]:
    config_vector_enabled = False
    config_vector_top_k = 8
    config_mode = "standard"
    config_intensive_chunk_tokens = 8192
    config_intensive_max_selected_excerpts = 12
    project = None
    if args.config is not None:
        project = load_project_config(args.config)
        config_vector_enabled = project.generation.vector_enabled
        config_vector_top_k = project.generation.vector_top_k
        config_mode = project.generation.ask_mode_default
        config_intensive_chunk_tokens = project.generation.intensive_chunk_tokens
        config_intensive_max_selected_excerpts = (
            project.generation.intensive_max_selected_excerpts
        )

    resolved_mode = cast(AskMode, args.mode or config_mode)
    if resolved_mode == "intensive" and project is None:
        raise EngLLMError(
            "--mode intensive requires --config so repository scope is known"
        )

    resolved_vector_enabled = (
        config_vector_enabled if args.vector_enabled is None else args.vector_enabled
    )
    resolved_vector_top_k = (
        config_vector_top_k if args.vector_top_k is None else args.vector_top_k
    )
    resolved_intensive_chunk_tokens = (
        config_intensive_chunk_tokens
        if args.intensive_chunk_tokens is None
        else args.intensive_chunk_tokens
    )
    resolved_intensive_max_selected_excerpts = (
        config_intensive_max_selected_excerpts
        if args.intensive_max_selected_excerpts is None
        else args.intensive_max_selected_excerpts
    )

    if resolved_vector_top_k <= 0:
        raise EngLLMError("--vector-top-k must be a positive integer")
    if resolved_intensive_chunk_tokens <= 0:
        raise EngLLMError("--intensive-chunk-tokens must be a positive integer")
    if resolved_intensive_max_selected_excerpts <= 0:
        raise EngLLMError(
            "--intensive-max-selected-excerpts must be a positive integer"
        )

    return (
        project,
        resolved_mode,
        resolved_vector_enabled,
        resolved_vector_top_k,
        resolved_intensive_chunk_tokens,
        resolved_intensive_max_selected_excerpts,
    )


def _build_ask_client(args: argparse.Namespace) -> LLMClient:
    resolved_temperature = _resolve_temperature(args.temperature, 0.2)
    return create_llm_client(
        LLMConfig(
            provider=args.provider,
            model_name=args.model,
            temperature=resolved_temperature,
        )
    )


def _run_ask_answer(args: argparse.Namespace) -> int:
    resolved_temperature = _resolve_temperature(args.temperature, 0.2)
    if args.graph_top_k <= 0:
        raise EngLLMError("--graph-top-k must be a positive integer")
    (
        project,
        resolved_mode,
        resolved_vector_enabled,
        resolved_vector_top_k,
        resolved_intensive_chunk_tokens,
        resolved_intensive_max_selected_excerpts,
    ) = _resolve_ask_settings(args)
    llm_client = _build_ask_client(args)
    request = QueryRequest(question=args.question, top_k=args.top_k)
    result = answer_question(
        request=request,
        index_path=args.index_path,
        llm_client=llm_client,
        model_name=args.model,
        temperature=resolved_temperature,
        mode=resolved_mode,
        project_config=project,
        repo_root=args.repo_root.resolve(),
        graph_enabled=not args.no_graph,
        graph_depth=args.graph_depth,
        graph_top_k=args.graph_top_k,
        vector_enabled=resolved_vector_enabled,
        vector_top_k=resolved_vector_top_k,
        intensive_chunk_tokens=resolved_intensive_chunk_tokens,
        intensive_max_selected_excerpts=resolved_intensive_max_selected_excerpts,
        progress_callback=_progress,
    )
    print(render_query_answer_text(result.answer))
    return 0


def _run_ask_interactive(args: argparse.Namespace) -> int:
    resolved_temperature = _resolve_temperature(args.temperature, 0.2)
    if args.graph_top_k <= 0:
        raise EngLLMError("--graph-top-k must be a positive integer")
    (
        project,
        resolved_mode,
        resolved_vector_enabled,
        resolved_vector_top_k,
        resolved_intensive_chunk_tokens,
        resolved_intensive_max_selected_excerpts,
    ) = _resolve_ask_settings(args)
    llm_client = _build_ask_client(args)

    history: list[str] = []
    while True:
        question = input("engllm ask> ").strip()
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
            mode=resolved_mode,
            project_config=project,
            repo_root=args.repo_root.resolve(),
            graph_enabled=not args.no_graph,
            graph_depth=args.graph_depth,
            graph_top_k=args.graph_top_k,
            vector_enabled=resolved_vector_enabled,
            vector_top_k=resolved_vector_top_k,
            intensive_chunk_tokens=resolved_intensive_chunk_tokens,
            intensive_max_selected_excerpts=resolved_intensive_max_selected_excerpts,
            progress_callback=_progress,
        )
        print(render_query_answer_text(result.answer))
        history.extend([f"Q: {question}", f"A: {result.answer.answer}"])
    return 0


def _run_repo_inspect_diff(args: argparse.Namespace) -> int:
    result = inspect_diff(
        commit_range=args.commit_range,
        repo_root=args.repo_root.resolve(),
    )
    print(result.impact.model_dump_json(indent=2))
    return 0


def _run_repo_migrate_index(args: argparse.Namespace) -> int:
    migrated_path = migrate_legacy_index(
        index_path=args.index_path,
        shard_size=args.shard_size,
        write_batch_size=args.write_batch_size,
        max_in_memory_records=args.max_in_memory_records,
    )
    print(f"Migrated retrieval index to sharded store: {migrated_path}")
    return 0


def _sdd_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="sdd",
        help="Software design description generation and updates.",
        commands=(
            ToolCommand(
                name="validate-config",
                help="Validate toolkit config, SDD targets, and template.",
                add_arguments=_add_sdd_common_args,
                run=_run_sdd_validate_config,
            ),
            ToolCommand(
                name="generate",
                help="Generate an initial SDD from repository evidence.",
                add_arguments=_add_sdd_common_args,
                run=_run_sdd_generate,
            ),
            ToolCommand(
                name="propose-updates",
                help="Generate commit-driven update proposals for an existing SDD.",
                add_arguments=_add_sdd_propose_args,
                run=_run_sdd_propose_updates,
            ),
        ),
    )


def _ask_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="ask",
        help="Question-answering workflows over repository evidence.",
        commands=(
            ToolCommand(
                name="answer",
                help="Answer one grounded question.",
                add_arguments=_add_ask_answer_args,
                run=_run_ask_answer,
            ),
            ToolCommand(
                name="interactive",
                help="Start an interactive grounded Q&A session.",
                add_arguments=_add_ask_common_args,
                run=_run_ask_interactive,
            ),
        ),
    )


def _repo_tool_spec() -> ToolSpec:
    return ToolSpec(
        name="repo",
        help="Repository utility commands.",
        commands=(
            ToolCommand(
                name="inspect-diff",
                help="Inspect one commit range and emit structured impact data.",
                add_arguments=_add_repo_inspect_diff_args,
                run=_run_repo_inspect_diff,
            ),
            ToolCommand(
                name="migrate-index",
                help="Migrate a legacy retrieval index into the sharded store format.",
                add_arguments=_add_repo_migrate_index_args,
                run=_run_repo_migrate_index,
            ),
        ),
    )


def _tool_specs() -> tuple[ToolSpec, ...]:
    return (_sdd_tool_spec(), _ask_tool_spec(), _repo_tool_spec())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="engllm")
    tool_parsers = parser.add_subparsers(dest="tool", required=True)

    for tool in _tool_specs():
        tool_parser = tool_parsers.add_parser(tool.name, help=tool.help)
        command_parsers = tool_parser.add_subparsers(dest="command", required=True)
        for command in tool.commands:
            command_parser = command_parsers.add_parser(command.name, help=command.help)
            command.add_arguments(command_parser)
            command_parser.set_defaults(_runner=command.run)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    try:
        args = _parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        runner = cast(
            Callable[[argparse.Namespace], int] | None, getattr(args, "_runner", None)
        )
        if runner is None:
            raise EngLLMError(
                f"Unknown command namespace: {args.tool} {getattr(args, 'command', '')}".strip()
            )
        return runner(args)
    except EngLLMError as exc:
        print(f"Error: {exc}")
        return 2

    print(json.dumps({"error": f"Unknown command {args.tool} {args.command}"}))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
