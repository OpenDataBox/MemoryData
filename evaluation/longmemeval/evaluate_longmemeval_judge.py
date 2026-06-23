"""Run LongMemEval official-style LLM-as-Judge on existing result files.

This script is intentionally post-processing only:
- it never modifies the source *_results.json files
- it writes a dedicated .longmemeval_judge.json sidecar next to each input file
- it prefers eval_metadata.question_id and falls back to legacy query matching when absent
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.longmemeval.longmemeval_judge import (
    expand_result_paths,
    process_longmemeval_result_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing LongMemEval result files with an official-style "
            "LLM judge and write dedicated sidecar outputs."
        )
    )
    parser.add_argument(
        "result_files",
        nargs="*",
        help="Existing *_results.json files to judge.",
    )
    parser.add_argument(
        "--result_glob",
        action="append",
        default=[],
        help="Glob pattern for result files. Can be provided multiple times.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o",
        help="OpenAI-compatible judge model name. Default: gpt-4o",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the judge API key.",
    )
    parser.add_argument(
        "--base-url-env",
        default="OPENAI_BASE_URL",
        help="Environment variable containing the OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Explicit OpenAI-compatible base URL. Overrides --base-url-env when set.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent judge calls. Default: 8",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of independent judge runs per row. Default: 1",
    )
    parser.add_argument(
        "--dataset-json",
        default=None,
        help="Optional explicit LongMemEval dataset JSON to use for canonical question lookup.",
    )
    parser.add_argument(
        "--sidecar-suffix",
        default=None,
        help=(
            "Optional filesystem-safe suffix to append to the sidecar filename, "
            "e.g. elephant_alpha -> *.longmemeval_judge.elephant_alpha.json"
        ),
    )
    parser.add_argument(
        "--sidecar-path",
        default=None,
        help="Optional explicit sidecar output path. Only valid when judging one result file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing sidecar and regenerate it from scratch.",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    result_paths = expand_result_paths(args.result_files, args.result_glob)
    if not result_paths:
        raise SystemExit("No result files provided. Use positional paths or --result_glob.")
    if args.sidecar_path and len(result_paths) != 1:
        raise SystemExit("--sidecar-path can only be used with exactly one result file.")

    print("LongMemEval judge summary")
    for path in result_paths:
        summary = await process_longmemeval_result_file(
            result_path=path,
            judge_model=args.judge_model,
            api_key_env=args.api_key_env,
            base_url_env=args.base_url_env,
            base_url=args.base_url,
            max_concurrency=args.max_concurrency,
            num_runs=args.num_runs,
            dataset_json=args.dataset_json,
            sidecar_suffix=args.sidecar_suffix,
            sidecar_path=args.sidecar_path,
            overwrite=args.overwrite,
        )

        print(f"- {summary['path']}")
        if summary.get("skipped"):
            print(f"  skipped: {summary.get('reason')}")
            if "sidecar_path" in summary:
                print(f"  sidecar_path: {summary['sidecar_path']}")
            if "total_rows" in summary:
                print(
                    "  total_rows={total_rows} supported_rows={supported_rows} "
                    "failed_rows_skipped={failed_rows_skipped} unsupported_rows={unsupported_rows}".format(
                        total_rows=summary.get("total_rows", 0),
                        supported_rows=summary.get("supported_rows", 0),
                        failed_rows_skipped=summary.get("failed_rows_skipped", 0),
                        unsupported_rows=summary.get("unsupported_rows", 0),
                    )
                )
            continue

        print(
            "  written={written} judged_rows={judged_rows}/{supported_rows} "
            "failed_rows_skipped={failed_rows_skipped} unsupported_rows={unsupported_rows} "
            "judge_accuracy={judge_accuracy} judge_accuracy_std={judge_accuracy_std}".format(
                written=summary.get("written", False),
                judged_rows=summary.get("judged_rows", 0),
                supported_rows=summary.get("supported_rows", 0),
                failed_rows_skipped=summary.get("failed_rows_skipped", 0),
                unsupported_rows=summary.get("unsupported_rows", 0),
                judge_accuracy=summary.get("judge_accuracy"),
                judge_accuracy_std=summary.get("judge_accuracy_std"),
            )
        )
        print(f"  sidecar_path: {summary.get('sidecar_path')}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
