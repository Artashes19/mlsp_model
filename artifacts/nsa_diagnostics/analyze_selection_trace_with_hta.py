#!/usr/bin/env python
"""
Analyze a selection-path Kineto trace with Holistic Trace Analysis (HTA).

Outputs a compact JSON/text summary focused on:
- temporal breakdown
- GPU kernel breakdown
- frequent CUDA kernel sequences
- optional trace diff if a second trace is provided
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from hta.trace_analysis import TraceAnalysis
from hta.trace_diff import DeviceType, TraceDiff


def df_records(df: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    if df is None:
        return []
    if len(df) > limit:
        df = df.head(limit)
    return json.loads(df.to_json(orient="records"))


def serialize_exception(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def analyze_single_trace(trace_dir: Path) -> dict[str, object]:
    ta = TraceAnalysis(trace_dir=str(trace_dir))

    temporal = None
    temporal_error = None
    try:
        temporal = ta.get_temporal_breakdown(visualize=False)
    except Exception as exc:  # pragma: no cover - HTA failure path
        temporal_error = serialize_exception(exc)

    kernel_type_df = None
    all_kernel_df = None
    kernel_breakdown_error = None
    try:
        kernel_type_df, all_kernel_df = ta.get_gpu_kernel_breakdown(
            visualize=False,
            include_memory_kernels=True,
        )
    except Exception as exc:  # pragma: no cover - HTA failure path
        kernel_breakdown_error = serialize_exception(exc)

    frequent = None
    frequent_error = None
    try:
        frequent = ta.get_frequent_cuda_kernel_sequences(
            operator_name="selection_forward",
            output_dir=str(trace_dir),
            rank=0,
            top_k=5,
            visualize=False,
        )
    except Exception as exc:  # pragma: no cover - HTA failure path
        frequent_error = serialize_exception(exc)

    return {
        "trace_dir": str(trace_dir),
        "temporal_breakdown_top": df_records(temporal, limit=8),
        "gpu_kernel_type_breakdown": df_records(kernel_type_df, limit=8),
        "gpu_kernel_breakdown_top": df_records(all_kernel_df, limit=12),
        "frequent_cuda_kernel_sequences": df_records(frequent, limit=10),
        "temporal_breakdown_error": temporal_error,
        "gpu_kernel_breakdown_error": kernel_breakdown_error,
        "frequent_cuda_kernel_sequences_error": frequent_error,
    }


def analyze_diff(control_dir: Path, test_dir: Path) -> dict[str, object]:
    diff_df = TraceDiff.compare_traces(
        control=str(control_dir),
        test=str(test_dir),
        device_type=DeviceType.GPU,
        use_short_name=True,
    )
    return {
        "control_trace_dir": str(control_dir),
        "test_trace_dir": str(test_dir),
        "trace_diff_top": df_records(diff_df, limit=20),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to one Kineto trace json or a directory containing only that trace",
    )
    parser.add_argument(
        "--control-trace",
        type=Path,
        help="Optional control Kineto trace json or trace directory for HTA diff",
    )
    parser.add_argument(
        "--test-trace",
        type=Path,
        help="Optional test Kineto trace json or trace directory for HTA diff",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def materialize_trace_dir(trace_path: Path, stack: ExitStack) -> Path:
    if trace_path.is_dir():
        return trace_path
    if trace_path.is_file():
        tmpdir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="hta_trace_")))
        shutil.copy2(trace_path, tmpdir / trace_path.name)
        return tmpdir
    raise SystemExit(f"Trace path does not exist: {trace_path}")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        trace_dir = materialize_trace_dir(args.trace, stack)
        result: dict[str, object] = {
            "timestamp": timestamp,
            "single_trace_analysis": analyze_single_trace(trace_dir),
        }

        if args.control_trace and args.test_trace:
            result["trace_diff_analysis"] = analyze_diff(
                materialize_trace_dir(args.control_trace, stack),
                materialize_trace_dir(args.test_trace, stack),
            )

        json_path = args.out_dir / f"selection_trace_hta_summary_{timestamp}.json"
        txt_path = args.out_dir / f"selection_trace_hta_summary_{timestamp}.txt"
        json_path.write_text(json.dumps(result, indent=2))

        lines = [
            f"Trace dir: {trace_dir}",
            f"Saved JSON: {json_path}",
        ]
        if "trace_diff_analysis" in result:
            lines.append("Trace diff included: yes")
        else:
            lines.append("Trace diff included: no")
        txt_path.write_text("\n".join(lines) + "\n")
        print(f"Saved JSON: {json_path}")
        print(f"Saved text: {txt_path}")


if __name__ == "__main__":
    main()
