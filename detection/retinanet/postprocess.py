import logging
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)


def generate_lung_mask(
    image_np: np.ndarray,
    thresh_val: float = 0.47,
    method: str = "slice",
) -> np.ndarray:
    """
    Generate a coarse lung mask in model-space array coordinates.

    The default method mirrors the n8n preprocessing idea, but is optimized for
    evaluation-time filtering: it thresholds air and cleans each axial slice,
    then keeps the largest lung-like air components on that slice. It avoids
    full-volume 3D fill/label operations that are too expensive inside detector
    evaluation loops.

    Args:
        image_np: 3D model-space CT array, usually normalized to [0, 1].
        thresh_val: air threshold in the same scale as image_np. For the
            default LUNA HU window [-1024, 300], HU -400 maps to about 0.47.
        method: "slice" for n8n-style slice-wise mask, "volume" for the
            previous full-volume connected-component mask.
    """
    if image_np.ndim != 3:
        raise ValueError(f"generate_lung_mask expects a 3D array, got shape={image_np.shape}")

    method = str(method or "slice").lower()
    if method in {"slice", "n8n", "n8n_style"}:
        return _generate_slice_lung_mask(image_np, thresh_val=thresh_val)
    if method in {"volume", "legacy"}:
        return _generate_volume_lung_mask(image_np, thresh_val=thresh_val)
    raise ValueError(f"Unsupported lung mask method: {method}")


def _generate_slice_lung_mask(image_np: np.ndarray, thresh_val: float) -> np.ndarray:
    start_t = time.time()
    air = image_np.astype(np.float32, copy=False) < float(thresh_val)
    lung = np.zeros_like(air, dtype=bool)
    structure_2d = np.ones((3, 3), dtype=bool)

    # Model-space arrays use [H, W, D]. Iterate over D to process axial slices.
    for z in range(air.shape[2]):
        slice_air = ndimage.binary_opening(air[:, :, z], structure=structure_2d, iterations=1)
        labels, _ = ndimage.label(slice_air)
        if labels.max() == 0:
            continue

        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        border_labels = border_labels[border_labels > 0]
        internal_air = slice_air & ~np.isin(labels, border_labels)
        internal_air = ndimage.binary_fill_holes(internal_air)
        internal_air = ndimage.binary_closing(internal_air, structure=structure_2d, iterations=2)
        internal_labels, num_internal = ndimage.label(internal_air)
        if num_internal == 0:
            continue
        counts = np.bincount(internal_labels.ravel())
        counts[0] = 0
        keep_count = min(2, int((counts > 0).sum()))
        keep_labels = np.argsort(counts)[-keep_count:]
        slice_lung = np.isin(internal_labels, keep_labels)
        slice_lung = ndimage.binary_closing(slice_lung, structure=structure_2d, iterations=1)
        lung[:, :, z] = slice_lung

    elapsed = time.time() - start_t
    logger.debug(
        "Slice lung mask generated in %.1fms. Mask fraction: %.1f%%",
        elapsed * 1000.0,
        100.0 * float(lung.sum()) / max(int(lung.size), 1),
    )
    return lung.astype(bool, copy=False)


def _generate_volume_lung_mask(image_np: np.ndarray, thresh_val: float) -> np.ndarray:
    start_t = time.time()
    binary = image_np < float(thresh_val)

    labeled_array, num_features = ndimage.label(binary)
    if num_features == 0:
        return np.zeros_like(binary, dtype=bool)

    border_labels = {
        labeled_array[0, 0, 0],
        labeled_array[0, 0, -1],
        labeled_array[0, -1, 0],
        labeled_array[0, -1, -1],
        labeled_array[-1, 0, 0],
        labeled_array[-1, 0, -1],
        labeled_array[-1, -1, 0],
        labeled_array[-1, -1, -1],
    }

    internal_air_mask = np.copy(binary)
    for bg_label in border_labels:
        if bg_label != 0:
            internal_air_mask[labeled_array == bg_label] = 0

    labeled_internal, num_internal = ndimage.label(internal_air_mask)
    if num_internal == 0:
        return np.zeros_like(binary, dtype=bool)

    component_sizes = np.bincount(labeled_internal.ravel())
    component_sizes[0] = 0
    top_2_labels = np.argsort(component_sizes)[::-1][:2]

    lung_mask = np.zeros_like(binary, dtype=bool)
    for label in top_2_labels:
        if component_sizes[label] > 5000:
            lung_mask[labeled_internal == label] = True

    struct = ndimage.generate_binary_structure(3, 1)
    lung_mask = ndimage.binary_dilation(lung_mask, structure=struct, iterations=7)
    lung_mask = ndimage.binary_closing(lung_mask, structure=struct, iterations=3)

    elapsed = time.time() - start_t
    logger.debug(
        "Volume lung mask generated in %.1fms. Mask fraction: %.1f%%",
        elapsed * 1000.0,
        100.0 * float(lung_mask.sum()) / max(int(lung_mask.size), 1),
    )
    return lung_mask.astype(bool, copy=False)


def box_lung_mask_metrics(box: Iterable[float], lung_mask: np.ndarray) -> Dict[str, float]:
    values = list(box)
    if len(values) != 6 or lung_mask.ndim != 3:
        return {"center_in_lung": False, "overlap_ratio": 0.0, "valid_box": False}

    h1, w1, d1, h2, w2, d2 = [float(v) for v in values]
    h_min, h_max = sorted([h1, h2])
    w_min, w_max = sorted([w1, w2])
    d_min, d_max = sorted([d1, d2])

    height, width, depth = lung_mask.shape
    hi0 = max(0, int(np.floor(h_min)))
    wi0 = max(0, int(np.floor(w_min)))
    di0 = max(0, int(np.floor(d_min)))
    hi1 = min(height - 1, int(np.ceil(h_max)))
    wi1 = min(width - 1, int(np.ceil(w_max)))
    di1 = min(depth - 1, int(np.ceil(d_max)))

    if hi0 > hi1 or wi0 > wi1 or di0 > di1:
        return {"center_in_lung": False, "overlap_ratio": 0.0, "valid_box": False}

    hc = min(height - 1, max(0, int(round((h_min + h_max) / 2.0))))
    wc = min(width - 1, max(0, int(round((w_min + w_max) / 2.0))))
    dc = min(depth - 1, max(0, int(round((d_min + d_max) / 2.0))))
    center_in_lung = bool(lung_mask[hc, wc, dc])

    roi = lung_mask[hi0 : hi1 + 1, wi0 : wi1 + 1, di0 : di1 + 1]
    overlap_ratio = float(np.mean(roi > 0)) if roi.size else 0.0
    return {
        "center_in_lung": center_in_lung,
        "overlap_ratio": overlap_ratio,
        "valid_box": True,
    }


def lung_mask_keep_flags(
    boxes: np.ndarray,
    lung_mask: np.ndarray,
    min_overlap_ratio: float = 0.01,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    metrics = [box_lung_mask_metrics(box, lung_mask) for box in boxes]
    keep = np.asarray(
        [
            bool(item["center_in_lung"]) or float(item["overlap_ratio"]) >= float(min_overlap_ratio)
            for item in metrics
        ],
        dtype=bool,
    )
    return keep, metrics
