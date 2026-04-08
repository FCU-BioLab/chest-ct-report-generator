#!/usr/bin/env python3
"""
Lightweight interactive viewer for FPR patch datasets.

Usage:
    python -m detection.retinanet.view_fpr_dataset --dataset_dir detection/results/fpr_dataset_xxx
"""

from __future__ import annotations

import argparse
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np


FILENAME_RE = re.compile(
    r"^(?P<split>.+?)_(?P<scan>.+?)_pred(?P<pred>\d+)_iou(?P<iou>-?\d+(?:\.\d+)?)_s(?P<score>-?\d+(?:\.\d+)?)\.npy$"
)


HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>FPR Dataset Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffaf2;
      --ink: #1f2430;
      --muted: #6b7280;
      --line: #d8cbb8;
      --accent: #b85c38;
      --accent-2: #2e6f95;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Noto Sans TC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff7df 0, transparent 28%),
        radial-gradient(circle at bottom right, #e7efe8 0, transparent 24%),
        var(--bg);
    }
    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: rgba(255, 250, 242, 0.92);
      padding: 20px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      backdrop-filter: blur(6px);
    }
    .main {
      padding: 20px;
    }
    h1, h2, h3 { margin: 0 0 12px 0; }
    h1 { font-size: 24px; }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .group { margin-bottom: 18px; }
    label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; }
    input, select, button {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 10px;
      font-size: 14px;
    }
    input[type=range] { padding: 0; }
    button {
      background: linear-gradient(135deg, var(--accent), #d97757);
      color: white;
      border: 0;
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary {
      background: white;
      color: var(--ink);
      border: 1px solid var(--line);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }
    .stat {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }
    .stat .k { font-size: 12px; color: var(--muted); }
    .stat .v { font-size: 22px; font-weight: 800; margin-top: 6px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 14px;
    }
    .card {
      background: rgba(255,255,255,0.9);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease;
    }
    .card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
    .card.active { outline: 2px solid var(--accent); }
    .meta-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      background: #f1ece2;
    }
    .pill.pos { background: #d9f1db; }
    .pill.neg { background: #fde2e2; }
    .card canvas {
      width: 100%;
      image-rendering: pixelated;
      border-radius: 10px;
      background: #0f172a;
      border: 1px solid #d6d6d6;
      aspect-ratio: 1 / 1;
    }
    .detail {
      background: rgba(255,255,255,0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      margin-bottom: 18px;
    }
    .viewer-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .viewer-panel {
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
    }
    .viewer-panel canvas {
      width: 100%;
      image-rendering: pixelated;
      background: #0f172a;
      border-radius: 10px;
      aspect-ratio: 1 / 1;
    }
    .controls-inline {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .pagination {
      display: flex;
      gap: 8px;
      align-items: center;
      margin: 18px 0;
      flex-wrap: wrap;
    }
    .pagination button { width: auto; padding: 8px 14px; }
    .jsonbox {
      font-family: Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      max-height: 240px;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
      .viewer-grid, .controls-inline { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>FPR Dataset Viewer</h1>
      <div class="hint" id="datasetInfo">loading...</div>

      <div class="stats">
        <div class="stat"><div class="k">Total</div><div class="v" id="statTotal">-</div></div>
        <div class="stat"><div class="k">Filtered</div><div class="v" id="statFiltered">-</div></div>
        <div class="stat"><div class="k">Positive</div><div class="v" id="statPos">-</div></div>
        <div class="stat"><div class="k">Negative</div><div class="v" id="statNeg">-</div></div>
      </div>

      <div class="group">
        <label for="labelFilter">Label</label>
        <select id="labelFilter">
          <option value="">All</option>
          <option value="positive">Positive</option>
          <option value="negative">Negative</option>
          <option value="ignored">Ignored</option>
        </select>
      </div>

      <div class="group">
        <label for="searchBox">Search scan / filename</label>
        <input id="searchBox" placeholder="seriesuid, filename, split...">
      </div>

      <div class="group">
        <label for="scoreMin">Score min</label>
        <input id="scoreMin" type="number" step="0.01" value="">
      </div>

      <div class="group">
        <label for="scoreMax">Score max</label>
        <input id="scoreMax" type="number" step="0.01" value="">
      </div>

      <div class="group">
        <label for="iouMin">IoU min</label>
        <input id="iouMin" type="number" step="0.01" value="">
      </div>

      <div class="group">
        <label for="iouMax">IoU max</label>
        <input id="iouMax" type="number" step="0.01" value="">
      </div>

      <div class="group">
        <label for="sortBy">Sort by</label>
        <select id="sortBy">
          <option value="score_desc">Score desc</option>
          <option value="score_asc">Score asc</option>
          <option value="iou_desc">IoU desc</option>
          <option value="iou_asc">IoU asc</option>
          <option value="scan_id">Scan ID</option>
          <option value="label">Label</option>
        </select>
      </div>

      <div class="group">
        <label for="perPage">Per page</label>
        <select id="perPage">
          <option>12</option>
          <option selected>24</option>
          <option>48</option>
          <option>96</option>
        </select>
      </div>

      <div class="group">
        <button id="applyBtn">Apply Filters</button>
      </div>
      <div class="group">
        <button class="secondary" id="resetBtn">Reset</button>
      </div>
    </aside>

    <main class="main">
      <section class="detail" id="detailSection">
        <h2 id="detailTitle">Select a sample</h2>
        <div class="hint" id="detailHint">Click any card below to inspect tri-planar slices and metadata.</div>
        <div class="controls-inline">
          <div>
            <label for="viewMode">View mode</label>
            <select id="viewMode">
              <option value="slice">Slice</option>
              <option value="mip">MIP</option>
            </select>
          </div>
          <div>
            <label for="sliceZ">Axial slice</label>
            <input id="sliceZ" type="range" min="0" max="31" value="16">
          </div>
          <div>
            <label for="sliceY">Coronal/Sagittal slice</label>
            <input id="sliceY" type="range" min="0" max="31" value="16">
          </div>
        </div>
        <div class="viewer-grid">
          <div class="viewer-panel">
            <h3>Axial (Y-X @ Z)</h3>
            <canvas id="canvasAxial" width="256" height="256"></canvas>
          </div>
          <div class="viewer-panel">
            <h3>Coronal (Y-Z @ X)</h3>
            <canvas id="canvasCoronal" width="256" height="256"></canvas>
          </div>
          <div class="viewer-panel">
            <h3>Sagittal (X-Z @ Y)</h3>
            <canvas id="canvasSagittal" width="256" height="256"></canvas>
          </div>
        </div>
        <div style="margin-top:12px" class="jsonbox" id="metaBox"></div>
      </section>

      <div class="pagination">
        <button id="prevBtn">Prev</button>
        <div id="pageInfo">Page 1</div>
        <button id="nextBtn">Next</button>
      </div>

      <section class="cards" id="cards"></section>
    </main>
  </div>

  <script>
    let currentPage = 1;
    let currentTotalPages = 1;
    let currentSelection = null;
    let currentSamples = [];
    let currentPatch = null;
    let datasetSummary = null;

    const qs = (id) => document.getElementById(id);

    function renderGrayCanvas(canvas, pixels, width, height) {
      const ctx = canvas.getContext('2d');
      const tmp = document.createElement('canvas');
      tmp.width = width;
      tmp.height = height;
      const tctx = tmp.getContext('2d');
      const imageData = tctx.createImageData(width, height);
      for (let i = 0; i < pixels.length; i++) {
        const v = pixels[i];
        const offset = i * 4;
        imageData.data[offset] = v;
        imageData.data[offset + 1] = v;
        imageData.data[offset + 2] = v;
        imageData.data[offset + 3] = 255;
      }
      tctx.putImageData(imageData, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    }

    function buildQuery(pageOverride) {
      const params = new URLSearchParams();
      params.set('page', pageOverride || currentPage);
      params.set('per_page', qs('perPage').value);
      if (qs('labelFilter').value) params.set('label', qs('labelFilter').value);
      if (qs('searchBox').value) params.set('search', qs('searchBox').value.trim());
      if (qs('scoreMin').value) params.set('score_min', qs('scoreMin').value);
      if (qs('scoreMax').value) params.set('score_max', qs('scoreMax').value);
      if (qs('iouMin').value) params.set('iou_min', qs('iouMin').value);
      if (qs('iouMax').value) params.set('iou_max', qs('iouMax').value);
      params.set('sort_by', qs('sortBy').value);
      return params.toString();
    }

    async function loadSummary() {
      const res = await fetch('/api/summary');
      datasetSummary = await res.json();
      qs('datasetInfo').textContent =
        `${datasetSummary.dataset_dir} | samples=${datasetSummary.total_samples} | patch=${datasetSummary.patch_shape.join('x')} | metadata=${datasetSummary.has_metadata ? 'yes' : 'fallback'}`;
      qs('statTotal').textContent = datasetSummary.total_samples;
      qs('statPos').textContent = datasetSummary.counts.positive || 0;
      qs('statNeg').textContent = datasetSummary.counts.negative || 0;
    }

    async function loadSamples(pageOverride) {
      if (pageOverride) currentPage = pageOverride;
      const res = await fetch(`/api/samples?${buildQuery(pageOverride)}`);
      const payload = await res.json();
      currentSamples = payload.items;
      currentPage = payload.page;
      currentTotalPages = payload.total_pages;
      qs('statFiltered').textContent = payload.total_filtered;
      qs('pageInfo').textContent = `Page ${payload.page} / ${payload.total_pages}`;

      const cards = qs('cards');
      cards.innerHTML = '';
      for (const item of payload.items) {
        const card = document.createElement('article');
        card.className = 'card';
        card.dataset.id = item.id;
        if (currentSelection && currentSelection.id === item.id) card.classList.add('active');
        card.innerHTML = `
          <canvas width="96" height="96"></canvas>
          <div class="meta-row">
            <span class="pill ${item.label === 'positive' ? 'pos' : 'neg'}">${item.label}</span>
            <span class="pill">score ${item.score_display}</span>
            <span class="pill">IoU ${item.iou_display}</span>
          </div>
          <div style="margin-top:8px; font-weight:700; word-break:break-all">${item.scan_id}</div>
          <div class="hint" style="margin-top:6px">${item.filename}</div>
        `;
        card.addEventListener('click', () => selectSample(item));
        cards.appendChild(card);

        const previewRes = await fetch(`/api/patch/${item.id}?axis=z&mode=slice&index=16`);
        const preview = await previewRes.json();
        renderGrayCanvas(card.querySelector('canvas'), preview.pixels, preview.width, preview.height);
      }
    }

    async function selectSample(item) {
      currentSelection = item;
      [...document.querySelectorAll('.card')].forEach(el => el.classList.toggle('active', Number(el.dataset.id) === item.id));
      qs('detailTitle').textContent = `${item.label} | ${item.scan_id}`;
      qs('detailHint').textContent = `${item.filename} | score=${item.score_display} | iou=${item.iou_display}`;

      const mode = qs('viewMode').value;
      const z = Number(qs('sliceZ').value);
      const y = Number(qs('sliceY').value);
      const [axialRes, coronalRes, sagittalRes, metaRes] = await Promise.all([
        fetch(`/api/patch/${item.id}?axis=z&mode=${mode}&index=${z}`),
        fetch(`/api/patch/${item.id}?axis=x&mode=${mode}&index=${y}`),
        fetch(`/api/patch/${item.id}?axis=y&mode=${mode}&index=${y}`),
        fetch(`/api/sample/${item.id}`),
      ]);
      const axial = await axialRes.json();
      const coronal = await coronalRes.json();
      const sagittal = await sagittalRes.json();
      const meta = await metaRes.json();

      currentPatch = meta;
      qs('sliceZ').max = meta.shape[2] - 1;
      qs('sliceY').max = Math.max(meta.shape[0], meta.shape[1]) - 1;
      renderGrayCanvas(qs('canvasAxial'), axial.pixels, axial.width, axial.height);
      renderGrayCanvas(qs('canvasCoronal'), coronal.pixels, coronal.width, coronal.height);
      renderGrayCanvas(qs('canvasSagittal'), sagittal.pixels, sagittal.width, sagittal.height);
      qs('metaBox').textContent = JSON.stringify(meta, null, 2);
    }

    function resetFilters() {
      qs('labelFilter').value = '';
      qs('searchBox').value = '';
      qs('scoreMin').value = '';
      qs('scoreMax').value = '';
      qs('iouMin').value = '';
      qs('iouMax').value = '';
      qs('sortBy').value = 'score_desc';
      qs('perPage').value = '24';
      currentPage = 1;
      loadSamples(1);
    }

    qs('applyBtn').addEventListener('click', () => loadSamples(1));
    qs('resetBtn').addEventListener('click', resetFilters);
    qs('prevBtn').addEventListener('click', () => { if (currentPage > 1) loadSamples(currentPage - 1); });
    qs('nextBtn').addEventListener('click', () => { if (currentPage < currentTotalPages) loadSamples(currentPage + 1); });
    qs('viewMode').addEventListener('change', () => currentSelection && selectSample(currentSelection));
    qs('sliceZ').addEventListener('input', () => currentSelection && selectSample(currentSelection));
    qs('sliceY').addEventListener('input', () => currentSelection && selectSample(currentSelection));

    loadSummary().then(() => loadSamples(1));
  </script>
</body>
</html>
"""


def load_samples(dataset_dir: Path) -> List[Dict]:
    metadata_path = dataset_dir / "metadata.json"
    samples: List[Dict] = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for idx, sample in enumerate(metadata.get("samples", [])):
            rel = sample.get("relative_path")
            path = dataset_dir / rel if rel else None
            sample = dict(sample)
            sample["id"] = idx
            sample["path"] = str(path) if path else None
            samples.append(sample)
        return samples

    idx = 0
    for label_dir in ("positive", "negative", "ignored"):
        root = dataset_dir / label_dir
        if not root.exists():
            continue
        for path in sorted(root.glob("*.npy")):
            match = FILENAME_RE.match(path.name)
            sample = {
                "id": idx,
                "filename": path.name,
                "relative_path": str(path.relative_to(dataset_dir)).replace("\\", "/"),
                "label": label_dir,
                "path": str(path),
            }
            if match:
                sample.update(
                    {
                        "source_split": match.group("split"),
                        "scan_id": match.group("scan"),
                        "pred_index": int(match.group("pred")),
                        "iou": float(match.group("iou")),
                        "score": float(match.group("score")),
                    }
                )
            else:
                sample.setdefault("scan_id", path.stem)
                sample.setdefault("score", None)
                sample.setdefault("iou", None)
            samples.append(sample)
            idx += 1
    return samples


def enrich_sample(sample: Dict, dataset_dir: Path) -> Dict:
    path = sample.get("path")
    if not path:
        rel = sample.get("relative_path")
        path = str(dataset_dir / rel) if rel else None
    s = dict(sample)
    s["path"] = path
    s.setdefault("scan_id", "unknown")
    s.setdefault("filename", Path(path).name if path else f"sample_{s['id']}")
    s["score_display"] = "-" if s.get("score") is None else f"{float(s['score']):.3f}"
    s["iou_display"] = "-" if s.get("iou") is None else f"{float(s['iou']):.3f}"
    return s


def build_summary(samples: List[Dict], dataset_dir: Path) -> Dict:
    counts: Dict[str, int] = {}
    patch_shape = [0, 0, 0]
    for sample in samples:
        label = sample.get("label", "unknown")
        counts[label] = counts.get(label, 0) + 1
    first_path = next((s.get("path") or str(dataset_dir / s.get("relative_path", "")) for s in samples if (s.get("path") or s.get("relative_path"))), None)
    if first_path and Path(first_path).exists():
        patch_shape = list(np.load(first_path).shape)
    return {
        "dataset_dir": str(dataset_dir),
        "total_samples": len(samples),
        "counts": counts,
        "patch_shape": patch_shape,
        "has_metadata": (dataset_dir / "metadata.json").exists(),
    }


def apply_filters(samples: List[Dict], params: Dict[str, List[str]]) -> List[Dict]:
    label = (params.get("label", [""])[0] or "").strip()
    search = (params.get("search", [""])[0] or "").strip().lower()
    score_min = _to_float(params.get("score_min", [""])[0])
    score_max = _to_float(params.get("score_max", [""])[0])
    iou_min = _to_float(params.get("iou_min", [""])[0])
    iou_max = _to_float(params.get("iou_max", [""])[0])
    sort_by = (params.get("sort_by", ["score_desc"])[0] or "score_desc").strip()

    items = []
    for sample in samples:
        if label and sample.get("label") != label:
            continue
        if search:
            hay = " ".join(
                str(sample.get(key, "")) for key in ("scan_id", "filename", "relative_path", "source_split", "label")
            ).lower()
            if search not in hay:
                continue
        score = sample.get("score")
        iou = sample.get("iou")
        if score_min is not None and (score is None or float(score) < score_min):
            continue
        if score_max is not None and (score is None or float(score) > score_max):
            continue
        if iou_min is not None and (iou is None or float(iou) < iou_min):
            continue
        if iou_max is not None and (iou is None or float(iou) > iou_max):
            continue
        items.append(sample)

    reverse = sort_by.endswith("_desc")
    if sort_by.startswith("score"):
        items.sort(key=lambda s: float(s.get("score") if s.get("score") is not None else -math.inf), reverse=reverse)
    elif sort_by.startswith("iou"):
        items.sort(key=lambda s: float(s.get("iou") if s.get("iou") is not None else -math.inf), reverse=reverse)
    elif sort_by == "scan_id":
        items.sort(key=lambda s: str(s.get("scan_id", "")))
    elif sort_by == "label":
        items.sort(key=lambda s: (str(s.get("label", "")), -float(s.get("score") or -math.inf)))
    return items


def _to_float(value: str) -> Optional[float]:
    try:
        value = str(value).strip()
        if not value:
            return None
        return float(value)
    except ValueError:
        return None


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "FPRViewer/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML)
            return
        if parsed.path == "/api/summary":
            self._send_json(self.server.app["summary"])
            return
        if parsed.path == "/api/samples":
            self._handle_samples(parsed)
            return
        if parsed.path.startswith("/api/sample/"):
            self._handle_sample(parsed)
            return
        if parsed.path.startswith("/api/patch/"):
            self._handle_patch(parsed)
            return
        self.send_error(404, "Not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _handle_samples(self, parsed) -> None:
        params = parse_qs(parsed.query)
        all_samples = self.server.app["samples"]
        filtered = apply_filters(all_samples, params)
        page = max(1, int(params.get("page", ["1"])[0]))
        per_page = max(1, min(200, int(params.get("per_page", ["24"])[0])))
        total_pages = max(1, math.ceil(len(filtered) / per_page))
        page = min(page, total_pages)
        start = (page - 1) * per_page
        end = start + per_page
        items = [enrich_sample(sample, self.server.app["dataset_dir"]) for sample in filtered[start:end]]
        self._send_json(
            {
                "page": page,
                "per_page": per_page,
                "total_filtered": len(filtered),
                "total_pages": total_pages,
                "items": items,
            }
        )

    def _handle_sample(self, parsed) -> None:
        sample_id = int(parsed.path.rsplit("/", 1)[-1])
        sample = enrich_sample(self.server.app["samples"][sample_id], self.server.app["dataset_dir"])
        arr = np.load(sample["path"])
        sample["shape"] = list(arr.shape)
        sample["stats"] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        self._send_json(sample)

    def _handle_patch(self, parsed) -> None:
        sample_id = int(parsed.path.rsplit("/", 1)[-1])
        params = parse_qs(parsed.query)
        axis = params.get("axis", ["z"])[0]
        mode = params.get("mode", ["slice"])[0]
        index = int(params.get("index", ["16"])[0])
        sample = enrich_sample(self.server.app["samples"][sample_id], self.server.app["dataset_dir"])
        arr = np.load(sample["path"]).astype(np.float32, copy=False)
        if mode == "mip":
            if axis == "z":
                view = arr.max(axis=2)
            elif axis == "x":
                view = arr.max(axis=1)
            elif axis == "y":
                view = arr.max(axis=0)
            else:
                view = arr.max(axis=2)
        else:
            if axis == "z":
                index = max(0, min(index, arr.shape[2] - 1))
                view = arr[:, :, index]
            elif axis == "x":
                index = max(0, min(index, arr.shape[1] - 1))
                view = arr[:, index, :]
            elif axis == "y":
                index = max(0, min(index, arr.shape[0] - 1))
                view = arr[index, :, :]
            else:
                index = max(0, min(index, arr.shape[2] - 1))
                view = arr[:, :, index]
        pixels = _normalize_to_uint8(view)
        self._send_json(
            {
                "width": int(view.shape[1]),
                "height": int(view.shape[0]),
                "pixels": pixels.reshape(-1).tolist(),
            }
        )

    def _send_json(self, payload: Dict) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_html(self, html: str) -> None:
        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def _normalize_to_uint8(view: np.ndarray) -> np.ndarray:
    lo = float(np.min(view))
    hi = float(np.max(view))
    if hi <= lo:
        return np.zeros_like(view, dtype=np.uint8)
    scaled = (view - lo) / (hi - lo)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive viewer for FPR patch datasets")
    parser.add_argument("--dataset_dir", required=True, help="FPR dataset directory")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind")
    parser.add_argument("--port", type=int, default=8765, help="port to bind")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    samples = load_samples(dataset_dir)
    if not samples:
        raise RuntimeError(f"No FPR samples found in {dataset_dir}")

    app = {
        "dataset_dir": dataset_dir,
        "samples": samples,
        "summary": build_summary(samples, dataset_dir),
    }
    httpd = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    httpd.app = app  # type: ignore[attr-defined]
    print(f"FPR viewer running at http://{args.host}:{args.port}")
    print(f"Dataset: {dataset_dir}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
