#!/usr/bin/env python3
"""
Per-case analysis export for RetinaNet evaluation.
"""

import html
import json
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .visualize_predictions import draw_boxes_on_slice

logger = logging.getLogger(__name__)


def _safe_case_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:120]


def _source_stem(path: Optional[str], fallback: str) -> str:
    if not path:
        return fallback
    p = Path(str(path))
    name = p.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = p.stem
    return _safe_case_name(name or fallback)


def _compute_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    iy1 = max(float(box_a[0]), float(box_b[0]))
    ix1 = max(float(box_a[1]), float(box_b[1]))
    iz1 = max(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    ix2 = min(float(box_a[4]), float(box_b[4]))
    iz2 = min(float(box_a[5]), float(box_b[5]))
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1) * max(0.0, iz2 - iz1)
    if inter <= 0.0:
        return 0.0
    vol_a = max(0.0, float(box_a[3]) - float(box_a[0])) * max(0.0, float(box_a[4]) - float(box_a[1])) * max(0.0, float(box_a[5]) - float(box_a[2]))
    vol_b = max(0.0, float(box_b[3]) - float(box_b[0])) * max(0.0, float(box_b[4]) - float(box_b[1])) * max(0.0, float(box_b[5]) - float(box_b[2]))
    return float(inter / max(vol_a + vol_b - inter, 1e-6))


def match_predictions(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_is_tp = np.zeros(len(pred_boxes), dtype=bool)
    pred_match_gt = np.full(len(pred_boxes), -1, dtype=np.int32)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return pred_is_tp, pred_match_gt, gt_matched
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt_box in enumerate(gt_boxes):
            iou = _compute_iou_3d(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
            pred_is_tp[i] = True
            pred_match_gt[i] = int(best_gt_idx)
            gt_matched[best_gt_idx] = True
    return pred_is_tp, pred_match_gt, gt_matched


def _boxes_to_mask(shape: Sequence[int], boxes: np.ndarray) -> np.ndarray:
    mask = np.zeros(tuple(int(v) for v in shape), dtype=np.uint16)
    for idx, box in enumerate(boxes, start=1):
        y1, x1, z1, y2, x2, z2 = [int(round(float(v))) for v in box]
        y1 = max(0, min(y1, mask.shape[0]))
        x1 = max(0, min(x1, mask.shape[1]))
        z1 = max(0, min(z1, mask.shape[2]))
        y2 = max(y1, min(y2, mask.shape[0]))
        x2 = max(x1, min(x2, mask.shape[1]))
        z2 = max(z1, min(z2, mask.shape[2]))
        if y2 > y1 and x2 > x1 and z2 > z1:
            mask[y1:y2, x1:x2, z1:z2] = idx
    return mask


def _save_nifti(array: np.ndarray, affine: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), str(path))


def _save_preview_png(frame: np.ndarray, path: Path) -> bool:
    try:
        from PIL import Image as PILImage

        path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(np.asarray(frame, dtype=np.uint8)).save(str(path))
        return True
    except Exception as exc:
        logger.warning("Failed to save preview PNG %s: %s", path, exc)
        return False


def _copy_original_ct_to_nifti(source_image: Optional[str], output_path: Path) -> Optional[str]:
    if not source_image:
        return None
    try:
        image = sitk.ReadImage(str(source_image))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_path))
        return None
    except Exception as exc:
        return str(exc)


def _choose_preview_slices(pred_boxes: np.ndarray, gt_boxes: np.ndarray, depth: int) -> List[int]:
    z_values: List[int] = []
    for box in list(pred_boxes) + list(gt_boxes):
        z_values.append(int(round((float(box[2]) + float(box[5])) / 2.0)))
    if not z_values:
        return [max(0, depth // 2)]
    z_values = sorted({max(0, min(depth - 1, z)) for z in z_values})
    if len(z_values) <= 6:
        return z_values
    idx = np.linspace(0, len(z_values) - 1, num=6, dtype=int)
    return [z_values[i] for i in idx]


def _best_iou_per_prediction(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    best = np.zeros((len(pred_boxes),), dtype=np.float32)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return best
    for i, pred_box in enumerate(pred_boxes):
        best[i] = max(_compute_iou_3d(pred_box, gt_box) for gt_box in gt_boxes)
    return best


def _compute_case_prf1(n_tp: int, n_fp: int, n_fn: int) -> Tuple[float, float, float]:
    precision = float(n_tp / (n_tp + n_fp)) if (n_tp + n_fp) > 0 else 0.0
    recall = float(n_tp / (n_tp + n_fn)) if (n_tp + n_fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _f1_bucket_name(summary: Dict) -> str:
    if bool(summary.get("is_tn_case", False)):
        return "tn_case"
    f1 = float(summary.get("case_f1", 0.0))
    if f1 < 0.20:
        return "f1_0.00_0.19"
    if f1 < 0.40:
        return "f1_0.20_0.39"
    if f1 < 0.60:
        return "f1_0.40_0.59"
    if f1 < 0.80:
        return "f1_0.60_0.79"
    return "f1_0.80_1.00"


def _prepare_fpr_trace(
    fpr_trace: Optional[Sequence[Dict]],
    gt_boxes: np.ndarray,
    iou_thresh: float,
    score_thresh: float,
) -> Tuple[List[Dict], Dict[str, int]]:
    if not fpr_trace:
        return [], {
            "n_fpr_candidates": 0,
            "n_fpr_kept": 0,
            "n_fpr_removed": 0,
            "n_fpr_removed_tp": 0,
            "n_fpr_removed_fp": 0,
            "n_fpr_kept_tp": 0,
            "n_fpr_kept_fp": 0,
            "n_fpr_final_kept": 0,
        }

    trace_boxes = np.asarray([item.get("box", [0, 0, 0, 0, 0, 0]) for item in fpr_trace], dtype=np.float32)
    pred_is_tp, pred_match_gt, _ = match_predictions(trace_boxes, gt_boxes, iou_thresh=iou_thresh)
    best_iou = _best_iou_per_prediction(trace_boxes, gt_boxes)

    prepared: List[Dict] = []
    counts = {
        "n_fpr_candidates": int(len(fpr_trace)),
        "n_fpr_kept": 0,
        "n_fpr_removed": 0,
        "n_fpr_removed_tp": 0,
        "n_fpr_removed_fp": 0,
        "n_fpr_kept_tp": 0,
        "n_fpr_kept_fp": 0,
        "n_fpr_final_kept": 0,
    }
    for i, item in enumerate(fpr_trace):
        row = dict(item)
        keep_after_fpr = bool(row.get("keep_after_fpr", True))
        final_score_after_fpr = float(row.get("final_score_after_fpr", row.get("det_score", 0.0)))
        keep_after_final_score = bool(final_score_after_fpr >= float(score_thresh)) if score_thresh is not None and score_thresh > 0 else keep_after_fpr
        candidate_is_tp = bool(pred_is_tp[i])
        row["candidate_is_tp"] = candidate_is_tp
        row["candidate_is_fp"] = bool(not candidate_is_tp)
        row["matched_gt_index"] = int(pred_match_gt[i])
        row["best_iou"] = float(best_iou[i])
        row["keep_after_final_score"] = keep_after_final_score
        prepared.append(row)

        if keep_after_fpr:
            counts["n_fpr_kept"] += 1
            counts["n_fpr_kept_tp"] += int(candidate_is_tp)
            counts["n_fpr_kept_fp"] += int(not candidate_is_tp)
        else:
            counts["n_fpr_removed"] += 1
            counts["n_fpr_removed_tp"] += int(candidate_is_tp)
            counts["n_fpr_removed_fp"] += int(not candidate_is_tp)
        counts["n_fpr_final_kept"] += int(keep_after_final_score)

    return prepared, counts


def _build_trace_rows(fpr_trace: Sequence[Dict]) -> str:
    rows = []
    for item in fpr_trace:
        rows.append(
            "<tr>"
            f"<td>{int(item.get('proposal_index', -1))}</td>"
            f"<td>{html.escape('TP' if item.get('candidate_is_tp') else 'FP')}</td>"
            f"<td>{float(item.get('best_iou', 0.0)):.3f}</td>"
            f"<td>{float(item.get('det_score', 0.0)):.3f}</td>"
            f"<td>{float(item.get('fpr_prob', 0.0)):.3f}</td>"
            f"<td>{float(item.get('final_score_after_fpr', 0.0)):.3f}</td>"
            f"<td>{html.escape(str(item.get('det_band', 'n/a')))}</td>"
            f"<td>{html.escape(str(item.get('applied_threshold', 'n/a')))}</td>"
            f"<td>{html.escape(str(bool(item.get('keep_after_fpr', True))))}</td>"
            f"<td>{html.escape(str(bool(item.get('keep_after_final_score', True))))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _build_niivue_overlay_controls(has_fpr_trace: bool) -> Tuple[List[Dict[str, object]], str]:
    overlay_defs: List[Dict[str, object]] = [
        {
            "file": "ct_model_space.nii.gz",
            "label": "CT (model space)",
            "colormap": "gray",
            "opacity": 1.0,
            "visible": True,
            "toggleable": False,
            "color_key": None,
        },
        {
            "file": "pred_tp_mask.nii.gz",
            "label": "Pred TP",
            "colormap": "maskGreen",
            "opacity": 0.45,
            "visible": True,
            "toggleable": True,
            "color_key": "maskGreen",
        },
        {
            "file": "pred_fp_mask.nii.gz",
            "label": "Pred FP",
            "colormap": "maskRed",
            "opacity": 0.45,
            "visible": True,
            "toggleable": True,
            "color_key": "maskRed",
        },
        {
            "file": "gt_fn_mask.nii.gz",
            "label": "GT FN",
            "colormap": "maskOrange",
            "opacity": 0.5,
            "visible": True,
            "toggleable": True,
            "color_key": "maskOrange",
        },
        {
            "file": "gt_mask.nii.gz",
            "label": "GT all",
            "colormap": "maskBlue",
            "opacity": 0.2,
            "visible": False,
            "toggleable": True,
            "color_key": "maskBlue",
        },
    ]
    if has_fpr_trace:
        overlay_defs.extend([
            {
                "file": "fpr_removed_tp_mask.nii.gz",
                "label": "FPR removed TP",
                "colormap": "maskMagenta",
                "opacity": 0.65,
                "visible": True,
                "toggleable": True,
                "color_key": "maskMagenta",
            },
            {
                "file": "fpr_removed_fp_mask.nii.gz",
                "label": "FPR removed FP",
                "colormap": "maskCyan",
                "opacity": 0.45,
                "visible": False,
                "toggleable": True,
                "color_key": "maskCyan",
            },
            {
                "file": "fpr_kept_fp_mask.nii.gz",
                "label": "FPR kept FP",
                "colormap": "maskYellow",
                "opacity": 0.55,
                "visible": False,
                "toggleable": True,
                "color_key": "maskYellow",
            },
        ])
    legend = []
    for idx, item in enumerate(overlay_defs):
        if not item["toggleable"]:
            legend.append(f'<div class="overlay-item base-volume">{html.escape(str(item["label"]))}</div>')
            continue
        checked = " checked" if item["visible"] else ""
        legend.append(
            f'<label class="overlay-item"><input type="checkbox" data-volume-index="{idx}"{checked}>'
            f'<span>{html.escape(str(item["label"]))}</span></label>'
        )
    return overlay_defs, "".join(legend)


def _encode_array_base64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _prepare_offline_viewer_payload(
    image_arr: np.ndarray,
    pred_tp_mask: np.ndarray,
    pred_fp_mask: np.ndarray,
    gt_fn_mask: np.ndarray,
    gt_mask: np.ndarray,
    fpr_removed_tp_mask: Optional[np.ndarray] = None,
    fpr_removed_fp_mask: Optional[np.ndarray] = None,
    fpr_kept_fp_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    image_f = np.asarray(image_arr, dtype=np.float32)
    vmin = float(np.min(image_f))
    vmax = float(np.max(image_f))
    if vmax - vmin < 1e-6:
        ct_uint8 = np.zeros_like(image_f, dtype=np.uint8)
    else:
        ct_uint8 = np.clip((image_f - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)

    overlay_bits = np.zeros(image_f.shape, dtype=np.uint8)
    overlays = [
        {"bit": 1, "label": "Pred TP", "color": [0, 255, 0], "visible": True, "source": pred_tp_mask},
        {"bit": 2, "label": "Pred FP", "color": [255, 64, 64], "visible": True, "source": pred_fp_mask},
        {"bit": 4, "label": "GT FN", "color": [255, 153, 0], "visible": True, "source": gt_fn_mask},
        {"bit": 8, "label": "GT all", "color": [64, 160, 255], "visible": False, "source": gt_mask},
    ]
    if fpr_removed_tp_mask is not None:
        overlays.append({"bit": 16, "label": "FPR removed TP", "color": [255, 64, 255], "visible": True, "source": fpr_removed_tp_mask})
    if fpr_removed_fp_mask is not None:
        overlays.append({"bit": 32, "label": "FPR removed FP", "color": [64, 255, 255], "visible": False, "source": fpr_removed_fp_mask})
    if fpr_kept_fp_mask is not None:
        overlays.append({"bit": 64, "label": "FPR kept FP", "color": [255, 255, 0], "visible": False, "source": fpr_kept_fp_mask})
    for item in overlays:
        source = np.asarray(item["source"])
        overlay_bits[source > 0] |= np.uint8(item["bit"])
        item.pop("source", None)

    return {
        "shape": [int(v) for v in image_f.shape],
        "ct_b64": _encode_array_base64(ct_uint8),
        "overlay_b64": _encode_array_base64(overlay_bits),
        "overlays": overlays,
    }


def _build_preview_html(
    case_name: str,
    summary: Dict,
    preview_files: List[str],
    has_fpr_trace: bool,
    viewer_payload: Dict[str, object],
) -> str:
    overlay_controls = "".join(
        f'<label class="overlay-item"><input type="checkbox" data-bit="{int(item["bit"])}"{" checked" if item.get("visible") else ""}>'
        f'<span>{html.escape(str(item["label"]))}</span></label>'
        for item in viewer_payload["overlays"]
    )
    viewer_json = json.dumps(
        {
            "shape": viewer_payload["shape"],
            "ct_b64": viewer_payload["ct_b64"],
            "overlay_b64": viewer_payload["overlay_b64"],
            "overlays": viewer_payload["overlays"],
        },
        ensure_ascii=False,
    )
    rows = []
    for key in (
        "source_image",
        "n_gt",
        "n_pred",
        "n_tp",
        "n_fp",
        "n_fn",
        "case_precision",
        "case_recall",
        "case_f1",
        "is_tn_case",
        "score_thresh",
        "iou_thresh",
    ):
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(summary.get(key)))}</td></tr>")
    if has_fpr_trace:
        for key in (
            "n_fpr_candidates",
            "n_fpr_kept",
            "n_fpr_removed",
            "n_fpr_removed_tp",
            "n_fpr_removed_fp",
            "n_fpr_kept_tp",
            "n_fpr_kept_fp",
            "n_fpr_final_kept",
        ):
            rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(summary.get(key)))}</td></tr>")
    previews = "".join(
        f'<div class="preview"><img src="{html.escape(name)}" alt="{html.escape(name)}"><p>{html.escape(name)}</p></div>'
        for name in preview_files
    )
    fpr_links = ""
    fpr_table = ""
    if has_fpr_trace:
        fpr_links = (
            ' | <a href="fpr_trace.json">fpr_trace.json</a>'
            ' | <a href="fpr_removed_tp_mask.nii.gz">fpr_removed_tp_mask.nii.gz</a>'
            ' | <a href="fpr_removed_fp_mask.nii.gz">fpr_removed_fp_mask.nii.gz</a>'
            ' | <a href="fpr_kept_tp_mask.nii.gz">fpr_kept_tp_mask.nii.gz</a>'
            ' | <a href="fpr_kept_fp_mask.nii.gz">fpr_kept_fp_mask.nii.gz</a>'
        )
        fpr_table = (
            "<h2>FPR Trace</h2>"
            "<table><tr><th>Idx</th><th>Candidate</th><th>Best IoU</th><th>Det</th><th>FPR</th><th>After FPR</th><th>Band</th><th>Threshold</th><th>Keep FPR</th><th>Keep Final</th></tr>"
            + _build_trace_rows(summary.get("fpr_trace", []))
            + "</table>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(case_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #11161c; color: #eef3f7; }}
    a {{ color: #8fd3ff; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #3a4653; padding: 6px 10px; text-align: left; }}
    .previews {{ display: flex; flex-wrap: wrap; gap: 16px; }}
    .preview img {{ max-width: 320px; border: 1px solid #3a4653; }}
    .viewer-shell {{ display: grid; grid-template-columns: minmax(280px, 360px) minmax(640px, 1fr); gap: 20px; align-items: start; margin-bottom: 24px; }}
    .panel {{ background: #171d24; border: 1px solid #2d3844; border-radius: 10px; padding: 14px; }}
    .panel h2 {{ margin: 0 0 12px 0; font-size: 18px; }}
    .overlay-list {{ display: grid; gap: 8px; margin-top: 12px; }}
    .overlay-item {{ display: flex; align-items: center; gap: 8px; }}
    .base-volume {{ font-weight: 600; color: #d9e4ef; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
    .toolbar button {{ background: #243241; color: #eef3f7; border: 1px solid #3a4653; border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
    .toolbar button:hover {{ background: #2d4154; }}
    #viewer-status {{ margin-top: 10px; color: #f7cf7e; font-size: 13px; }}
    #slice-canvas {{ width: 100%; max-width: 960px; image-rendering: pixelated; background: #000; border-radius: 10px; border: 1px solid #2d3844; }}
    .slice-controls {{ display: grid; gap: 10px; margin-top: 14px; }}
    #slice-slider {{ width: 100%; }}
    #slice-label {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>{html.escape(case_name)}</h1>
  <table>{"".join(rows)}</table>
  <p>
    Files: <a href="ct_model_space.nii.gz">ct_model_space.nii.gz</a> |
    <a href="ct_original.nii.gz">ct_original.nii.gz</a> |
    <a href="pred_tp_mask.nii.gz">pred_tp_mask.nii.gz</a> |
    <a href="pred_fp_mask.nii.gz">pred_fp_mask.nii.gz</a> |
    <a href="gt_mask.nii.gz">gt_mask.nii.gz</a> |
    <a href="gt_fn_mask.nii.gz">gt_fn_mask.nii.gz</a> |
    <a href="summary.json">summary.json</a>{fpr_links}
  </p>
  <div class="viewer-shell">
    <section class="panel">
      <h2>Offline Viewer</h2>
      <p>這個版本不依賴外網。它不是 WebGL 3D render，而是可互動的 slice viewer。要抓 FPR 問題，優先看 <code>FPR removed TP</code> 和 <code>FPR kept FP</code>。</p>
      <div class="toolbar">
        <button type="button" data-view="axial">Axial</button>
        <button type="button" data-view="coronal">Coronal</button>
        <button type="button" data-view="sagittal">Sagittal</button>
      </div>
      <div class="overlay-list">{overlay_controls}</div>
      <div class="slice-controls">
        <input id="slice-slider" type="range" min="0" max="0" value="0">
        <div id="slice-label"></div>
      </div>
      <div id="viewer-status">載入中。</div>
    </section>
    <section class="panel">
      <canvas id="slice-canvas" width="768" height="768"></canvas>
    </section>
  </div>
  <div class="previews">{previews}</div>
  {fpr_table}
  <script>
    const viewerData = {viewer_json};
    const statusEl = document.getElementById("viewer-status");
    const canvas = document.getElementById("slice-canvas");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("slice-slider");
    const sliceLabel = document.getElementById("slice-label");
    const [dimY, dimX, dimZ] = viewerData.shape;

    function decodeBase64Uint8(b64) {{
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }}

    const ct = decodeBase64Uint8(viewerData.ct_b64);
    const overlay = decodeBase64Uint8(viewerData.overlay_b64);
    const overlayVisible = new Map(viewerData.overlays.map((item) => [item.bit, !!item.visible]));
    let currentView = "axial";
    let currentIndex = 0;

    function dimsForView(view) {{
      if (view === "axial") return {{ width: dimX, height: dimY, depth: dimZ }};
      if (view === "coronal") return {{ width: dimY, height: dimZ, depth: dimX }};
      return {{ width: dimX, height: dimZ, depth: dimY }};
    }}

    function voxelIndex(y, x, z) {{
      return ((y * dimX) + x) * dimZ + z;
    }}

    function sampleView(view, row, col, slice) {{
      if (view === "axial") return voxelIndex(row, col, slice);
      if (view === "coronal") return voxelIndex(col, slice, row);
      return voxelIndex(slice, col, row);
    }}

    function redraw() {{
      const dims = dimsForView(currentView);
      const imageData = ctx.createImageData(dims.width, dims.height);
      for (let row = 0; row < dims.height; row += 1) {{
        const srcRow = currentView === "axial" ? row : dims.height - 1 - row;
        for (let col = 0; col < dims.width; col += 1) {{
          const srcIdx = sampleView(currentView, srcRow, col, currentIndex);
          const pix = ct[srcIdx];
          let r = pix, g = pix, b = pix;
          const bits = overlay[srcIdx];
          for (const item of viewerData.overlays) {{
            if (!overlayVisible.get(item.bit)) continue;
            if ((bits & item.bit) === 0) continue;
            r = Math.round(r * 0.35 + item.color[0] * 0.65);
            g = Math.round(g * 0.35 + item.color[1] * 0.65);
            b = Math.round(b * 0.35 + item.color[2] * 0.65);
          }}
          const out = (row * dims.width + col) * 4;
          imageData.data[out] = r;
          imageData.data[out + 1] = g;
          imageData.data[out + 2] = b;
          imageData.data[out + 3] = 255;
        }}
      }}
      canvas.width = dims.width;
      canvas.height = dims.height;
      ctx.putImageData(imageData, 0, 0);
      sliceLabel.textContent = `${{currentView}} slice ${{currentIndex + 1}} / ${{dims.depth}}`;
      statusEl.textContent = "離線 viewer 已載入。切到 GT FN、FPR removed TP、FPR kept FP 交叉比對。";
    }}

    function setView(view) {{
      currentView = view;
      const dims = dimsForView(view);
      slider.max = String(Math.max(0, dims.depth - 1));
      currentIndex = Math.min(currentIndex, dims.depth - 1);
      slider.value = String(currentIndex);
      redraw();
    }}

    slider.addEventListener("input", () => {{
      currentIndex = Number(slider.value || 0);
      redraw();
    }});
    document.querySelectorAll("[data-view]").forEach((button) => {{
      button.addEventListener("click", () => setView(button.getAttribute("data-view")));
    }});
    document.querySelectorAll("[data-bit]").forEach((input) => {{
      input.addEventListener("change", (event) => {{
        overlayVisible.set(Number(event.target.getAttribute("data-bit")), !!event.target.checked);
        redraw();
      }});
    }});
    setView("axial");
  </script>
</body>
</html>
"""


def export_case_analysis(
    *,
    image_yxz: np.ndarray,
    affine: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    output_root: Path,
    source_image: Optional[str],
    fallback_case_id: str,
    iou_thresh: float,
    score_thresh: float,
    fpr_trace: Optional[Sequence[Dict]] = None,
) -> Dict:
    case_name = _source_stem(source_image, fallback_case_id)
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    if score_thresh is not None and score_thresh > 0 and len(pred_scores) > 0:
        keep = pred_scores >= float(score_thresh)
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

    pred_is_tp, pred_match_gt, gt_matched = match_predictions(pred_boxes, gt_boxes, iou_thresh=iou_thresh)
    pred_best_iou = _best_iou_per_prediction(pred_boxes, gt_boxes)
    pred_tp = pred_boxes[pred_is_tp]
    pred_fp = pred_boxes[~pred_is_tp] if len(pred_boxes) > 0 else np.zeros((0, 6), dtype=np.float32)
    gt_fn = gt_boxes[~gt_matched] if len(gt_boxes) > 0 else np.zeros((0, 6), dtype=np.float32)
    prepared_fpr_trace, fpr_counts = _prepare_fpr_trace(fpr_trace, gt_boxes, iou_thresh=iou_thresh, score_thresh=score_thresh)

    image_arr = np.asarray(image_yxz, dtype=np.float32)
    affine_arr = np.asarray(affine, dtype=np.float32)
    if affine_arr.shape != (4, 4):
        affine_arr = np.eye(4, dtype=np.float32)

    _save_nifti(image_arr, affine_arr, case_dir / "ct_model_space.nii.gz")
    _save_nifti(_boxes_to_mask(image_arr.shape, pred_boxes), affine_arr, case_dir / "pred_all_mask.nii.gz")
    pred_tp_mask = _boxes_to_mask(image_arr.shape, pred_tp)
    pred_fp_mask = _boxes_to_mask(image_arr.shape, pred_fp)
    gt_mask = _boxes_to_mask(image_arr.shape, gt_boxes)
    gt_fn_mask = _boxes_to_mask(image_arr.shape, gt_fn)
    _save_nifti(pred_tp_mask, affine_arr, case_dir / "pred_tp_mask.nii.gz")
    _save_nifti(pred_fp_mask, affine_arr, case_dir / "pred_fp_mask.nii.gz")
    _save_nifti(gt_mask, affine_arr, case_dir / "gt_mask.nii.gz")
    _save_nifti(gt_fn_mask, affine_arr, case_dir / "gt_fn_mask.nii.gz")
    fpr_removed_tp_mask = None
    fpr_removed_fp_mask = None
    fpr_kept_fp_mask = None
    if prepared_fpr_trace:
        trace_boxes = np.asarray([item["box"] for item in prepared_fpr_trace], dtype=np.float32)
        trace_keep = np.asarray([bool(item["keep_after_fpr"]) for item in prepared_fpr_trace], dtype=bool)
        trace_tp = np.asarray([bool(item["candidate_is_tp"]) for item in prepared_fpr_trace], dtype=bool)
        fpr_removed_tp_mask = _boxes_to_mask(image_arr.shape, trace_boxes[(~trace_keep) & trace_tp])
        fpr_removed_fp_mask = _boxes_to_mask(image_arr.shape, trace_boxes[(~trace_keep) & (~trace_tp)])
        fpr_kept_tp_mask = _boxes_to_mask(image_arr.shape, trace_boxes[trace_keep & trace_tp])
        fpr_kept_fp_mask = _boxes_to_mask(image_arr.shape, trace_boxes[trace_keep & (~trace_tp)])
        _save_nifti(fpr_removed_tp_mask, affine_arr, case_dir / "fpr_removed_tp_mask.nii.gz")
        _save_nifti(fpr_removed_fp_mask, affine_arr, case_dir / "fpr_removed_fp_mask.nii.gz")
        _save_nifti(fpr_kept_tp_mask, affine_arr, case_dir / "fpr_kept_tp_mask.nii.gz")
        _save_nifti(fpr_kept_fp_mask, affine_arr, case_dir / "fpr_kept_fp_mask.nii.gz")

    original_copy_error = _copy_original_ct_to_nifti(source_image, case_dir / "ct_original.nii.gz")

    preview_files: List[str] = []
    for z_idx in _choose_preview_slices(pred_boxes, gt_boxes, image_arr.shape[2]):
        frame = draw_boxes_on_slice(
            image_arr[:, :, z_idx],
            pred_boxes,
            pred_scores,
            z_idx,
            is_tp=pred_is_tp,
            nodule_metrics={"TP": int(np.sum(pred_is_tp)), "FP": int(np.sum(~pred_is_tp)), "FN": int(np.sum(~gt_matched))},
            gt_boxes=gt_boxes,
            gt_is_matched=gt_matched,
        )
        png_name = f"slice_z{z_idx:03d}.png"
        if _save_preview_png(frame, case_dir / png_name):
            preview_files.append(png_name)

    summary = {
        "case_name": case_name,
        "source_image": source_image,
        "n_gt": int(len(gt_boxes)),
        "n_pred": int(len(pred_boxes)),
        "n_tp": int(np.sum(pred_is_tp)),
        "n_fp": int(np.sum(~pred_is_tp)) if len(pred_boxes) > 0 else 0,
        "n_fn": int(np.sum(~gt_matched)) if len(gt_boxes) > 0 else 0,
        "is_tn_case": bool(len(gt_boxes) == 0 and len(pred_boxes) == 0),
        "score_thresh": float(score_thresh) if score_thresh is not None else None,
        "iou_thresh": float(iou_thresh),
        "pred_scores": [float(v) for v in pred_scores.tolist()],
        "pred_boxes_yxz": [[float(x) for x in row] for row in pred_boxes.tolist()],
        "pred_is_tp": [bool(v) for v in pred_is_tp.tolist()],
        "pred_best_iou": [float(v) for v in pred_best_iou.tolist()],
        "pred_match_gt": [int(v) for v in pred_match_gt.tolist()],
        "gt_boxes_yxz": [[float(x) for x in row] for row in gt_boxes.tolist()],
        "original_copy_error": original_copy_error,
    }
    case_precision, case_recall, case_f1 = _compute_case_prf1(
        n_tp=int(summary["n_tp"]),
        n_fp=int(summary["n_fp"]),
        n_fn=int(summary["n_fn"]),
    )
    summary["case_precision"] = case_precision
    summary["case_recall"] = case_recall
    summary["case_f1"] = case_f1
    summary["f1_bucket"] = _f1_bucket_name(summary)
    summary.update(fpr_counts)
    if prepared_fpr_trace:
        summary["fpr_trace"] = prepared_fpr_trace
    with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if prepared_fpr_trace:
        with open(case_dir / "fpr_trace.json", "w", encoding="utf-8") as f:
            json.dump(prepared_fpr_trace, f, indent=2, ensure_ascii=False)
    viewer_payload = _prepare_offline_viewer_payload(
        image_arr=image_arr,
        pred_tp_mask=pred_tp_mask,
        pred_fp_mask=pred_fp_mask,
        gt_fn_mask=gt_fn_mask,
        gt_mask=gt_mask,
        fpr_removed_tp_mask=fpr_removed_tp_mask,
        fpr_removed_fp_mask=fpr_removed_fp_mask,
        fpr_kept_fp_mask=fpr_kept_fp_mask,
    )
    with open(case_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(_build_preview_html(case_name, summary, preview_files, has_fpr_trace=bool(prepared_fpr_trace), viewer_payload=viewer_payload))
    return summary


def write_case_analysis_index(output_root: Path, summaries: Sequence[Dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in summaries:
        case_name = str(summary.get("case_name"))
        rows.append(
            "<tr>"
            f'<td><a href="{html.escape(case_name)}/index.html">{html.escape(case_name)}</a></td>'
            f"<td>{float(summary.get('case_f1', 0.0)):.4f}</td>"
            f"<td>{float(summary.get('case_precision', 0.0)):.4f}</td>"
            f"<td>{float(summary.get('case_recall', 0.0)):.4f}</td>"
            f"<td>{int(summary.get('n_gt', 0))}</td>"
            f"<td>{int(summary.get('n_pred', 0))}</td>"
            f"<td>{int(summary.get('n_tp', 0))}</td>"
            f"<td>{int(summary.get('n_fp', 0))}</td>"
            f"<td>{int(summary.get('n_fn', 0))}</td>"
            f"<td>{html.escape(str(summary.get('f1_bucket', _f1_bucket_name(summary))))}</td>"
            f"<td>{html.escape(str(summary.get('is_tn_case', False)))}</td>"
            "</tr>"
        )
    with open(output_root / "index.html", "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Case Analysis</title>"
            "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}</style>"
            "</head><body><h1>Case Analysis</h1><p><a href=\"by_f1/index.html\">View grouped by case F1</a></p><table><tr><th>Case</th><th>F1</th><th>Precision</th><th>Recall</th><th>GT</th><th>Pred</th><th>TP</th><th>FP</th><th>FN</th><th>Bucket</th><th>TN Case</th></tr>"
            + "".join(rows)
            + "</table></body></html>"
        )
    with open(output_root / "cases.json", "w", encoding="utf-8") as f:
        json.dump(list(summaries), f, indent=2, ensure_ascii=False)

    by_f1_root = output_root / "by_f1"
    by_f1_root.mkdir(parents=True, exist_ok=True)
    bucket_map: Dict[str, List[Dict]] = {}
    for summary in summaries:
        bucket = _f1_bucket_name(summary)
        bucket_map.setdefault(bucket, []).append(summary)

    bucket_rows = []
    for bucket, items in sorted(bucket_map.items()):
        bucket_dir = by_f1_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        sorted_items = sorted(
            items,
            key=lambda x: (float(x.get("case_f1", 0.0)), int(x.get("n_fn", 0)), -int(x.get("n_fp", 0))),
        )
        rows = []
        for summary in sorted_items:
            case_name = str(summary.get("case_name"))
            rows.append(
                "<tr>"
                f'<td><a href="../../{html.escape(case_name)}/index.html">{html.escape(case_name)}</a></td>'
                f"<td>{float(summary.get('case_f1', 0.0)):.4f}</td>"
                f"<td>{float(summary.get('case_precision', 0.0)):.4f}</td>"
                f"<td>{float(summary.get('case_recall', 0.0)):.4f}</td>"
                f"<td>{int(summary.get('n_tp', 0))}</td>"
                f"<td>{int(summary.get('n_fp', 0))}</td>"
                f"<td>{int(summary.get('n_fn', 0))}</td>"
                "</tr>"
            )
        with open(bucket_dir / "cases.json", "w", encoding="utf-8") as f:
            json.dump(sorted_items, f, indent=2, ensure_ascii=False)
        with open(bucket_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(
                "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                f"<title>{html.escape(bucket)}</title>"
                "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}</style>"
                "</head><body>"
                f"<h1>{html.escape(bucket)}</h1>"
                "<p><a href=\"../index.html\">Back to F1 buckets</a></p>"
                "<table><tr><th>Case</th><th>F1</th><th>Precision</th><th>Recall</th><th>TP</th><th>FP</th><th>FN</th></tr>"
                + "".join(rows)
                + "</table></body></html>"
            )
        bucket_rows.append(
            "<tr>"
            f'<td><a href="{html.escape(bucket)}/index.html">{html.escape(bucket)}</a></td>'
            f"<td>{len(sorted_items)}</td>"
            f"<td>{float(np.mean([float(x.get('case_f1', 0.0)) for x in sorted_items])):.4f}</td>"
            f"<td>{int(sum(int(x.get('n_fn', 0)) for x in sorted_items))}</td>"
            "</tr>"
        )
    with open(by_f1_root / "index.html", "w", encoding="utf-8") as f:
        f.write(
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Cases by F1</title>"
            "<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;}</style>"
            "</head><body><h1>Cases by F1</h1>"
            "<p>Use this view to quickly find extreme low-F1 cases that may drag down overall performance.</p>"
            "<table><tr><th>Bucket</th><th>Cases</th><th>Mean F1</th><th>Total FN</th></tr>"
            + "".join(bucket_rows)
            + "</table></body></html>"
        )
