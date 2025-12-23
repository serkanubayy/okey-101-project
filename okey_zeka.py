import itertools
from collections import defaultdict

class OkeyZeka:
    def __init__(self):
        pass

    # =============================
    # PARSE
    # =============================
    def parse_tiles(self, detected_tiles):
        tiles = []
        for t in detected_tiles:
            if t.get("is_joker"):
                tiles.append({
                    "id": t["id"],
                    "color": t["color"],
                    "val": t["val"],
                    "joker": True
                })
            else:
                if t.get("val") is None:
                    continue
                tiles.append({
                    "id": t["id"],
                    "color": t["color"],
                    "val": t["val"],
                    "joker": False
                })
        return tiles

    # =============================
    # RUN (RENK SERİSİ)
    # =============================
    def find_runs(self, tiles):
        runs = []
        by_color = defaultdict(list)

        for t in tiles:
            if not t["joker"]:
                by_color[t["color"]].append(t)

        for color, lst in by_color.items():
            lst = sorted(lst, key=lambda x: x["val"])
            n = len(lst)

            for i in range(n):
                run = [lst[i]]
                last = lst[i]["val"]

                for j in range(i+1, n):
                    if lst[j]["val"] == last + 1:
                        run.append(lst[j])
                        last += 1
                        if len(run) >= 3:
                            runs.append(run.copy())
                    elif lst[j]["val"] == last:
                        continue
                    else:
                        break

        return runs

    # =============================
    # SET (AYNI SAYI FARKLI RENK)
    # =============================
    def find_sets(self, tiles):
        sets = []
        by_val = defaultdict(list)

        for t in tiles:
            if not t["joker"]:
                by_val[t["val"]].append(t)

        for val, lst in by_val.items():
            unique = {}
            for t in lst:
                if t["color"] not in unique:
                    unique[t["color"]] = t

            uniq = list(unique.values())

            if len(uniq) >= 3:
                sets.append(uniq)

            if len(uniq) == 4:
                for c in itertools.combinations(uniq, 3):
                    sets.append(list(c))

        return sets

    # =============================
    # ADAY PERLER
    # =============================
    def get_candidates(self, tiles):
        runs = self.find_runs(tiles)
        sets = self.find_sets(tiles)
        return runs + sets

    # =============================
    # PUAN
    # =============================
    def score_group(self, group):
        return sum(t["val"] for t in group if not t.get("joker"))

    # =============================
    # BACKTRACK (MAX SCORE)
    # =============================
    def backtrack(self, candidates, idx, used_ids):
        if idx >= len(candidates):
            return 0, []

        best_score, best_groups = self.backtrack(candidates, idx + 1, used_ids)

        group = candidates[idx]
        ids = {t["id"] for t in group}

        if ids & used_ids:
            return best_score, best_groups

        score_here = self.score_group(group)
        next_score, next_groups = self.backtrack(
            candidates,
            idx + 1,
            used_ids | ids
        )

        total = score_here + next_score

        if total > best_score:
            return total, [group] + next_groups

        return best_score, best_groups

    # =============================
    # ANA ÇAĞRI
    # =============================
    def find_best_hand(self, detected_tiles):
        tiles = self.parse_tiles(detected_tiles)

        candidates = self.get_candidates(tiles)
        candidates.sort(key=self.score_group, reverse=True)

        best_score, best_groups = self.backtrack(candidates, 0, set())

        return best_groups, best_score
