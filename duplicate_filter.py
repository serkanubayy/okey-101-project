from __future__ import annotations
from typing import List, Dict, Any
import math

def _xyxy(box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)

def box_area(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    ua = box_area(a) + box_area(b) - inter
    return 0.0 if ua <= 0 else inter / ua

def dist2(c1, c2):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return dx*dx + dy*dy

def suppress_duplicates(
    detected: List[Dict[str, Any]],
    center_dist_px: float = 30.0,
    iou_th: float = 0.35,
    area_ratio_th: float = 0.35,
) -> List[Dict[str, Any]]:

    normal = []
    passthrough = []

    for t in detected:
        b = _xyxy(t.get("box"))
        if b is None:
            passthrough.append(t)
            continue

        tt = dict(t)
        tt["_box"] = b
        tt["_area"] = box_area(b)
        tt["_center"] = box_center(b)
        tt["_conf"] = float(t.get("conf", 0.0))
        normal.append(tt)

    if len(normal) <= 1:
        return detected

    clusters: List[List[Dict[str, Any]]] = []
    dist2_th = center_dist_px * center_dist_px

    for t in normal:
        placed = False
        for cl in clusters:
            rep = cl[0]
            d2 = dist2(t["_center"], rep["_center"])
            if d2 > dist2_th:
                continue

            rep_iou = iou(t["_box"], rep["_box"])

            a1 = t["_area"]
            a2 = rep["_area"]
            ratio = abs(a1 - a2) / max(a1, a2) if (a1 > 0 and a2 > 0) else 1.0

            same_id = (t.get("id") is not None and rep.get("id") is not None and t.get("id") == rep.get("id"))

            if same_id or rep_iou >= iou_th or ratio <= area_ratio_th:
                cl.append(t)
                placed = True
                break

        if not placed:
            clusters.append([t])

    picked = []
    for cl in clusters:
        def score(x):
            c = (x.get("color") or "Unknown")
            v = x.get("val", None)

            good_color = 1.0 if (c != "Unknown") else 0.0
            good_num = 1.0 if (v is not None) else 0.0

            conf_part = x["_conf"] * 15.0
            area_part = math.sqrt(max(x["_area"], 1.0)) * 0.001
            quality_part = (good_color * 2.0) + (good_num * 3.0)

            return quality_part * 10.0 + conf_part + area_part

        best = max(cl, key=score)
        out = dict(best)
        out.pop("_box", None)
        out.pop("_area", None)
        out.pop("_center", None)
        out.pop("_conf", None)
        picked.append(out)

    # passthrough + picked
    return picked + passthrough
