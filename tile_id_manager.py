import numpy as np

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-9
    return inter / union

def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def _dist2(c1, c2):
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    return dx * dx + dy * dy

class TileIDManager:
    """
    Her frame'de gelen box listesine kalıcı ID atar.

    ✅ Bu sürüm:
    - reset() eklendi
    - assign_ids() artık hem:
        - [ [x1,y1,x2,y2], ... ]
        - [ {"box":[x1,y1,x2,y2], "conf":0.8}, ... ]
      formatlarını kabul eder.
    - return formatı: [(tile_id, box_xyxy, conf), ...]
    """

    def __init__(self, iou_th=0.4, dist_th=80.0, ttl=8):
        self.iou_th = float(iou_th)
        self.dist_th = float(dist_th)
        self.ttl = int(ttl)

        # prev: list of dict {id, box, center, miss, conf}
        self.prev = []
        self.next_id = 0

    def reset(self):
        self.prev = []
        self.next_id = 0

    def _prune(self):
        alive = []
        for p in self.prev:
            if p["miss"] <= self.ttl:
                alive.append(p)
        self.prev = alive

    def assign_ids(self, boxes_input):
        """
        boxes_input:
          - list of xyxy boxes
          - OR list of dict {"box":xyxy, "conf":float}
        returns:
          list of (id, xyxy, conf)
        """

        # normalize input
        cur_boxes = []
        cur_confs = []

        for item in boxes_input:
            if isinstance(item, dict):
                b = item.get("box", None)
                c = float(item.get("conf", 1.0))
            else:
                b = item
                c = 1.0

            if b is None:
                continue
            b = list(map(float, b))
            cur_boxes.append(b)
            cur_confs.append(float(c))

        # prev miss++
        for p in self.prev:
            p["miss"] += 1

        if len(cur_boxes) == 0:
            self._prune()
            return []

        pairs = []
        dist2_th = self.dist_th * self.dist_th

        for j, p in enumerate(self.prev):
            pb = p["box"]
            pc = p["center"]
            for i, b in enumerate(cur_boxes):
                bc = _center(b)
                d2 = _dist2(pc, bc)
                if d2 > dist2_th:
                    continue

                iouv = _iou(b, pb)
                dist_score = 1.0 - min(d2 / dist2_th, 1.0)
                score = (iouv * 2.0) + (dist_score * 0.5)

                if iouv >= (self.iou_th * 0.50) or dist_score >= 0.70:
                    pairs.append((score, j, i, iouv))

        pairs.sort(key=lambda x: x[0], reverse=True)

        used_prev = set()
        used_cur = set()
        assigned = [None] * len(cur_boxes)

        for score, j, i, iouv in pairs:
            if j in used_prev or i in used_cur:
                continue

            if iouv >= self.iou_th or score >= 1.10:
                used_prev.add(j)
                used_cur.add(i)

                pid = self.prev[j]["id"]
                b = cur_boxes[i]
                c = cur_confs[i]
                assigned[i] = (pid, b, c)

                self.prev[j]["box"] = b
                self.prev[j]["center"] = _center(b)
                self.prev[j]["miss"] = 0
                self.prev[j]["conf"] = c

        # unmatched -> new id
        for i, b in enumerate(cur_boxes):
            if assigned[i] is None:
                nid = self.next_id
                self.next_id += 1
                c = cur_confs[i]
                assigned[i] = (nid, b, c)
                self.prev.append({"id": nid, "box": b, "center": _center(b), "miss": 0, "conf": c})

        self._prune()
        return assigned
