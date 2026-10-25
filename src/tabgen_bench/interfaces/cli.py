"""Non-interactive CLI over the same public services as the Python API."""

import argparse
import json
import sys
import traceback
from pathlib import Path
from uuid import uuid4

from tabgen_bench import __version__
from tabgen_bench.core.errors import TabGenBenchError
from tabgen_bench.experiments.artifacts import ArtifactConflictError, IntegrityError
from tabgen_bench.reporting.records import IncompatibleRunsError

from . import api


def parser():
    root = argparse.ArgumentParser(
        prog="tabbench", description="TabGenBench: offline tabular evaluation"
    )
    root.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument("--debug", action="store_true", help="include exception traceback on stderr")
    commands = root.add_subparsers(dest="command")
    evaluate = commands.add_parser(
        "evaluate", help="evaluate external CSV/Parquet; no SDV required"
    )
    evaluate.add_argument("--config", required=True, type=Path)
    run = commands.add_parser("run", help="execute a frozen experiment")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true", help="read plan/cache only; no run records")
    for option in ("project-root", "data-root", "run-root", "cache-root"):
        run.add_argument(f"--{option}", type=Path)
    compare = commands.add_parser("compare", help="compare compatible persisted results")
    compare.add_argument("--runs", nargs="+", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--run-root", default=Path("runs"), type=Path)
    report = commands.add_parser("report", help="redraw persisted results without evaluation")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--run-root", default=Path("runs"), type=Path)
    datasets = commands.add_parser("datasets", help="manage pinned public sources")
    actions = datasets.add_subparsers(dest="dataset_action", required=True)
    for action in ("fetch", "verify"):
        sub = actions.add_parser(action)
        sub.add_argument("dataset_ids", nargs="*", help="default: all three frozen datasets")
        sub.add_argument("--project-root", type=Path)
        sub.add_argument("--data-root", type=Path)
    return root


def run_path(path, root):
    return path if path.exists() or path.parent != Path(".") else root / path


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    command_parser = parser()
    args = command_parser.parse_args(arguments)
    if args.command is None:
        command_parser.print_help()
        return 0
    try:
        if args.command == "evaluate":
            result = api.evaluate(args.config, command=("tabbench", *arguments))
            code = 3 if result["status"] in {"failed", "partial_failure"} else 0
        elif args.command == "run":
            result = api.run(
                args.config,
                dry_run=args.dry_run,
                project_root=args.project_root,
                data_root=args.data_root,
                run_root=args.run_root,
                cache_root=args.cache_root,
                command=("tabbench", *arguments),
            )
            code = (
                3
                if any(r["status"] in {"failed", "partial_failure"} for r in result.get("runs", []))
                else 0
            )
        elif args.command == "compare":
            result = api.compare(
                [run_path(p, args.run_root) for p in args.runs], output=args.output
            )
            code = 0
        elif args.command == "report":
            result = api.report(run_path(args.run, args.run_root), output=args.output)
            code = 0
        else:
            action = api.datasets_fetch if args.dataset_action == "fetch" else api.datasets_verify
            result = action(
                args.dataset_ids, project_root=args.project_root, data_root=args.data_root
            )
            code = 3 if any(r["status"] == "failed" for r in result["datasets"]) else 0
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return code
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        if isinstance(error, KeyboardInterrupt):
            code = 130
        elif isinstance(
            error, (IntegrityError, ArtifactConflictError, FileExistsError, IncompatibleRunsError)
        ):
            code = 4
        elif isinstance(error, (TabGenBenchError, ValueError, KeyError, OSError)):
            code = 2
        else:
            code = 1
        diagnostic = {
            "trace_id": str(uuid4()),
            "command": args.command,
            "arguments": arguments,
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", type(error).__name__),
            "error_message": str(error) or repr(error),
            "exit_code": code,
        }
        print(json.dumps(diagnostic, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return code


def app():
    raise SystemExit(main())


if __name__ == "__main__":
    app()
