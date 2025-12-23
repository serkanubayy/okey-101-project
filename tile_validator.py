from collections import defaultdict
from typing import List, Dict, Any, Tuple

MAX_DUPLICATE_PER_TILE = 2
MIN_NUMBER = 1
MAX_NUMBER = 13

VALID_COLORS = {"Red", "Blue", "Black", "Orange"}

def validate_tiles(detected: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings = []
    counter = defaultdict(list)

    for t in detected:
        color = t.get("color")
        val = t.get("val")

        if color not in VALID_COLORS:
            warnings.append(f"DROP: Invalid color {color}")
            continue

        if not isinstance(val, int) or not (MIN_NUMBER <= val <= MAX_NUMBER):
            warnings.append(f"DROP: Invalid number {val}")
            continue

        counter[(color, val)].append(t)

    valid_tiles = []
    for key, tiles in counter.items():
        if len(tiles) > MAX_DUPLICATE_PER_TILE:
            # En güvenilir 2 taneyi tut
            def score(x):
                # number_conf + color_conf ağırlıklı
                nc = float(x.get("number_conf", 0.0))
                cc = float(x.get("color_conf", 0.0))
                return (nc * 0.65) + (cc * 0.35)

            tiles_sorted = sorted(tiles, key=score, reverse=True)
            kept = tiles_sorted[:MAX_DUPLICATE_PER_TILE]
            dropped = tiles_sorted[MAX_DUPLICATE_PER_TILE:]

            valid_tiles.extend(kept)
            for d in dropped:
                warnings.append(f"DROP: Physical limit exceeded {key[0]}{key[1]}")
        else:
            valid_tiles.extend(tiles)

    return valid_tiles, warnings
