from collections import deque, Counter

class ColorStabilizer:
    def __init__(self, window=8, min_votes=3, lock_after=5):
        self.window = window
        self.min_votes = min_votes
        self.lock_after = lock_after

        self.history = {}   # tile_id -> deque
        self.locked = {}    # tile_id -> color

    def update(self, tile_id, color, conf=0.0):
        # Eğer kilitliyse direkt dön
        if tile_id in self.locked:
            return self.locked[tile_id]

        if tile_id not in self.history:
            self.history[tile_id] = deque(maxlen=self.window)

        # Unknown'ları düşük ağırlıkla ekle
        weight = 0.5 if color == "Unknown" else 1.0
        self.history[tile_id].append((color, weight))

        # Oyları topla
        counter = {}
        for c, w in self.history[tile_id]:
            counter[c] = counter.get(c, 0) + w

        if not counter:
            return color

        best_color, best_score = max(counter.items(), key=lambda x: x[1])

        # Kilitleme şartı
        if best_score >= self.min_votes and len(self.history[tile_id]) >= self.lock_after:
            self.locked[tile_id] = best_color
            return best_color

        return best_color
