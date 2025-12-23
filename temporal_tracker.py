# temporal_tracker.py
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple, Any


@dataclass
class TrackResult:
    number: Optional[int] = None
    number_conf: float = 0.0
    color: str = "Unknown"
    color_conf: float = 0.0


class TemporalTracker:
    """
    Her taş için son N frame/okuma boyunca gelen tahminleri tutar ve
    "en güvenilir" sonucu döndürür.

    - number: CNN çıktısı (1..13)
    - color: CNN/HSV çıktısı ("Red","Blue","Orange","Black","Unknown")
    """

    def __init__(
        self,
        window: int = 9,
        min_votes: int = 4,
        conf_floor: float = 0.25,
        lock_after: int = 6,
        lock_min_conf: float = 0.70,
    ):
        self.window = int(window)
        self.min_votes = int(min_votes)
        self.conf_floor = float(conf_floor)
        self.lock_after = int(lock_after)
        self.lock_min_conf = float(lock_min_conf)

        self._hist: Dict[Any, Deque[Tuple[Optional[int], float, str, float]]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

        self._locked: Dict[Any, TrackResult] = {}
        self._streak: Dict[Any, Tuple[Tuple[Optional[int], str], int]] = {}

    def reset(self, tile_id: Optional[Any] = None):
        if tile_id is None:
            self._hist.clear()
            self._locked.clear()
            self._streak.clear()
            return
        self._hist.pop(tile_id, None)
        self._locked.pop(tile_id, None)
        self._streak.pop(tile_id, None)

    def is_locked(self, tile_id: Any) -> bool:
        return tile_id in self._locked

    def update(
        self,
        tile_id: Any,
        number: Optional[int],
        number_conf: float,
        color: str,
        color_conf: float,
    ) -> TrackResult:

        if tile_id in self._locked:
            return self._locked[tile_id]

        if number is not None:
            try:
                number = int(number)
            except Exception:
                number = None

        try:
            number_conf = float(number_conf)
        except Exception:
            number_conf = 0.0

        try:
            color_conf = float(color_conf)
        except Exception:
            color_conf = 0.0

        color = (color or "Unknown").strip()
        if color == "":
            color = "Unknown"

        self._hist[tile_id].append((number, number_conf, color, color_conf))

        key = (number, color)
        last_key, streak = self._streak.get(tile_id, ((None, "Unknown"), 0))
        if key == last_key:
            streak += 1
        else:
            streak = 1
        self._streak[tile_id] = (key, streak)

        if (
            streak >= self.lock_after
            and number is not None
            and number_conf >= self.lock_min_conf
            and color != "Unknown"
            and color_conf >= (self.lock_min_conf * 0.85)
        ):
            locked = TrackResult(number=number, number_conf=number_conf, color=color, color_conf=color_conf)
            self._locked[tile_id] = locked
            return locked

        return self.get(tile_id)

    def get(self, tile_id: Any) -> TrackResult:
        if tile_id in self._locked:
            return self._locked[tile_id]

        hist = self._hist.get(tile_id)
        if not hist:
            return TrackResult()

        num_scores: Dict[int, float] = defaultdict(float)
        num_votes: Dict[int, int] = defaultdict(int)

        col_scores: Dict[str, float] = defaultdict(float)
        col_votes: Dict[str, int] = defaultdict(int)

        for (n, nconf, c, cconf) in hist:
            if n is not None and nconf >= self.conf_floor:
                num_scores[n] += float(nconf)
                num_votes[n] += 1

            # Unknown'a production cezası: oy sayma
            if c and c != "Unknown" and cconf >= self.conf_floor:
                col_scores[c] += float(cconf)
                col_votes[c] += 1

        best_num, best_num_conf = self._pick_best(num_scores, num_votes)
        best_col, best_col_conf = self._pick_best(col_scores, col_votes)

        if best_num is not None:
            if num_votes.get(best_num, 0) < self.min_votes:
                if best_num_conf < (self.lock_min_conf * 0.90):
                    best_num = None
                    best_num_conf = 0.0

        if best_col != "Unknown":
            if col_votes.get(best_col, 0) < self.min_votes:
                if best_col_conf < (self.lock_min_conf * 0.90):
                    best_col = "Unknown"
                    best_col_conf = 0.0

        if best_col is None:
            best_col = "Unknown"

        return TrackResult(
            number=best_num,
            number_conf=float(best_num_conf),
            color=str(best_col),
            color_conf=float(best_col_conf),
        )

    def summary_counts(self, track_results):
        """
        Main tarafında doğru READY/LOCK saymak için yardımcı.
        track_results: [(id, TrackResult), ...]
        """
        ready = 0
        locked = 0
        locked_color = 0
        locked_num = 0

        for tid, tr in track_results:
            if self.is_locked(tid):
                locked += 1
            if tr.color != "Unknown":
                locked_color += 1
            if tr.number is not None:
                locked_num += 1
            if tr.color != "Unknown" and tr.number is not None:
                ready += 1

        return {
            "ready": ready,
            "locked": locked,
            "locked_color": locked_color,
            "locked_num": locked_num,
        }

    @staticmethod
    def _pick_best(scores: Dict[Any, float], votes: Dict[Any, int]) -> Tuple[Optional[Any], float]:
        if not scores:
            return None, 0.0
        items = list(scores.items())
        items.sort(key=lambda kv: (kv[1], votes.get(kv[0], 0)), reverse=True)
        k, sc = items[0]
        return k, float(sc) / max(votes.get(k, 1), 1)
