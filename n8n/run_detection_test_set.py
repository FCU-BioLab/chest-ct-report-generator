"""Submit the detection test split to the n8n chest CT pipeline webhook.

This is a small client-side helper. n8n still orchestrates the actual stages:
preprocess -> detect -> segment -> feature -> report.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML_OUTPUT_DIR = Path("F:/chest-ct-report-output/n8n_detection_test_html")


def load_manifest_items(path: Path, split: str) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get(split)
    if not isinstance(items, list):
        raise ValueError(f"Manifest does not contain a list split named '{split}': {path}")
    return [item for item in items if isinstance(item, dict)]


def case_id_for(item: Dict[str, Any], index: int, prefix: str) -> str:
    series_uid = str(item.get("seriesuid") or item.get("id") or "").strip()
    if series_uid:
        safe_uid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in series_uid)
        return f"{prefix}{index:03d}_{safe_uid[-32:]}"
    return f"{prefix}{index:03d}"


def post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw}


def iter_selected(items: List[Dict[str, Any]], start: int, limit: int) -> Iterable[tuple[int, Dict[str, Any]]]:
    selected = items[start:]
    if limit > 0:
        selected = selected[:limit]
    for offset, item in enumerate(selected, start=start + 1):
        yield offset, item


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "case"


def copy_case_html(result: Dict[str, Any], html_output_dir: Path) -> str:
    if result.get("status") != "ok":
        return ""

    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    source = response.get("summary_html_path") or ""
    if not source:
        return ""

    source_path = Path(source)
    if not source_path.exists():
        result["html_copy_error"] = f"summary_html_path not found: {source_path}"
        return ""

    case_dir = html_output_dir / safe_filename(str(result.get("case_id", "case")))
    case_dir.mkdir(parents=True, exist_ok=True)
    target_path = case_dir / "summary.html"
    shutil.copy2(source_path, target_path)
    return str(target_path)


def rel_link(path: str, base_dir: Path) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(base_dir).as_posix()
    except ValueError:
        return p.as_posix()


def render_html_index(results: List[Dict[str, Any]], html_output_dir: Path, summary: Dict[str, Any]) -> str:
    html_output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        html_path = str(result.get("html_path") or "")
        html_cell = ""
        if html_path:
            link = html_lib.escape(rel_link(html_path, html_output_dir))
            html_cell = f'<a href="{link}">summary.html</a>'
        elif result.get("html_copy_error"):
            html_cell = html_lib.escape(str(result["html_copy_error"]))

        report_text_path = html_lib.escape(str(response.get("report_text_path") or ""))
        report_json_path = html_lib.escape(str(response.get("report_json_path") or ""))
        rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(result.get('index', '')))}</td>"
            f"<td>{html_lib.escape(str(result.get('case_id', '')))}</td>"
            f"<td>{html_lib.escape(str(result.get('status', '')))}</td>"
            f"<td>{html_lib.escape(str(result.get('elapsed_sec', '')))}</td>"
            f"<td>{html_lib.escape(str(response.get('nodule_count', '')))}</td>"
            f"<td>{html_lib.escape(str(response.get('report_generation_method', '')))}</td>"
            f"<td>{html_lib.escape(str(response.get('llm_validation_fix_count', '')))}</td>"
            f"<td>{html_cell}</td>"
            f"<td>{report_text_path}</td>"
            f"<td>{report_json_path}</td>"
            f"<td>{html_lib.escape(str(result.get('error', '')))}</td>"
            "</tr>"
        )

    summary_items = "".join(
        f"<dt>{html_lib.escape(str(key))}</dt><dd>{html_lib.escape(str(value))}</dd>"
        for key, value in summary.items()
    )
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Detection Test Set End-to-End Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    .meta {{ color: #52606d; margin-bottom: 20px; }}
    dl {{ display: grid; grid-template-columns: 180px 1fr; gap: 6px 12px; max-width: 900px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #f0f4f8; text-align: left; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    a {{ color: #0b69a3; }}
  </style>
</head>
<body>
  <h1>Detection Test Set End-to-End Results</h1>
  <div class="meta">Generated at {html_lib.escape(generated_at)}</div>
  <dl>{summary_items}</dl>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Case ID</th>
        <th>Status</th>
        <th>Elapsed sec</th>
        <th>Nodules</th>
        <th>Report Method</th>
        <th>LLM Fixes</th>
        <th>HTML</th>
        <th>Report TXT</th>
        <th>Report JSON</th>
        <th>Error</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def write_outputs(results: List[Dict[str, Any]], args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    args.html_output_dir.mkdir(parents=True, exist_ok=True)
    html_index_path = args.html_output_dir / "index.html"
    html_index_path.write_text(render_html_index(results, args.html_output_dir, summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webhook-url", default="http://localhost:5678/webhook/chest-ct-pipeline")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "detection/manifests/dataset_luna16_new_consensus.json")
    parser.add_argument("--split", default="testing")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--python-exe", default=str(PROJECT_ROOT / "venv/Scripts/python.exe"))
    parser.add_argument("--work-dir", default=str(PROJECT_ROOT / "n8n/runtime_detection_test"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--medsam2-checkpoint", default="")
    parser.add_argument("--llm-base-model", default=str(PROJECT_ROOT / "models/llm/Llama-3.2-1B-Instruct"))
    parser.add_argument("--llm-adapter", default=str(PROJECT_ROOT / "llm/ct_report_pipeline/assets/models/lora_ct_report/llama32_20260520"))
    parser.add_argument("--disable-llm", action="store_true")
    parser.add_argument("--disable-llm-validate", action="store_true")
    parser.add_argument("--no-propagate", action="store_true")
    parser.add_argument("--case-prefix", default="test-")
    parser.add_argument("--start", type=int, default=0, help="Zero-based item offset within the split")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining cases")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "n8n/runtime_detection_test/results.json")
    parser.add_argument("--html-output-dir", type=Path, default=DEFAULT_HTML_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_manifest_items(args.manifest, args.split)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for index, item in iter_selected(items, args.start, args.limit):
        input_path = item.get("image")
        if not input_path:
            results.append({"index": index, "status": "skipped", "error": "missing image path"})
            continue

        payload = {
            "case_id": case_id_for(item, index, args.case_prefix),
            "input_path": str(input_path),
            "model_path": str(args.model_path),
            "repo_root": str(args.repo_root),
            "python_exe": str(args.python_exe),
            "work_dir": str(args.work_dir),
            "threshold": args.threshold,
            "device": args.device,
            "medsam2_checkpoint": str(args.medsam2_checkpoint),
            "use_llm": not args.disable_llm,
            "llm_base_model": str(args.llm_base_model),
            "llm_adapter": str(args.llm_adapter),
            "llm_validate_output": not args.disable_llm_validate,
            "no_propagate": args.no_propagate,
        }

        started = time.time()
        try:
            response = post_json(args.webhook_url, payload, args.timeout_sec)
            elapsed = round(time.time() - started, 3)
            result = {
                "index": index,
                "status": "ok",
                "case_id": payload["case_id"],
                "input_path": payload["input_path"],
                "elapsed_sec": elapsed,
                "response": response,
            }
            result["html_path"] = copy_case_html(result, args.html_output_dir)
            print(json.dumps(result, ensure_ascii=False))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            result = {
                "index": index,
                "status": "failed",
                "case_id": payload["case_id"],
                "input_path": payload["input_path"],
                "error": str(exc),
            }
            print(json.dumps(result, ensure_ascii=False))
        results.append(result)
        partial_summary = {
            "manifest": str(args.manifest),
            "split": args.split,
            "submitted": len(results),
            "ok": sum(1 for item in results if item.get("status") == "ok"),
            "failed": sum(1 for item in results if item.get("status") == "failed"),
            "skipped": sum(1 for item in results if item.get("status") == "skipped"),
            "output_json": str(args.output_json),
            "html_index_path": str(args.html_output_dir / "index.html"),
        }
        write_outputs(results, args, partial_summary)

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    summary = {
        "manifest": str(args.manifest),
        "split": args.split,
        "submitted": len(results),
        "ok": sum(1 for item in results if item.get("status") == "ok"),
        "failed": sum(1 for item in results if item.get("status") == "failed"),
        "skipped": sum(1 for item in results if item.get("status") == "skipped"),
        "output_json": str(args.output_json),
        "html_index_path": str(args.html_output_dir / "index.html"),
        "html_output_dir": str(args.html_output_dir),
    }
    write_outputs(results, args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
