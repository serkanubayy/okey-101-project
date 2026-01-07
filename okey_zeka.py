import itertools
from collections import defaultdict

class OkeyZeka:
    def __init__(self):
        pass

    # ==========================================
    # 1. PARSE & JOKER AYRIŞTIRMA
    # ==========================================
    def parse_tiles(self, detected_tiles, indicator_tile):
        j_color = indicator_tile["color"]
        j_val = indicator_tile["val"] + 1
        if j_val > 13: j_val = 1
        
        # Taşları ve Jokerleri Ayır
        normal_tiles = []
        wildcards = [] 

        for t in detected_tiles:
            color = t["color"]
            val = t["val"]
            is_okey = (color == j_color and val == j_val)
            
            tile_obj = {
                "id": t.get("id"),
                "color": color,
                "val": val,
                "is_wildcard": is_okey, 
                "virtual_val": val,     
                "virtual_str": ""       
            }

            if is_okey:
                wildcards.append(tile_obj)
            else:
                normal_tiles.append(tile_obj)
        
        print(f"🧠 ANALİZ RAPORU -> GÖSTERGE: {indicator_tile['color']}{indicator_tile['val']} | JOKER: {j_color}{j_val}")
        print(f"👀 TESPİT EDİLEN JOKER SAYISI: {len(wildcards)}")
                
        return normal_tiles, wildcards, (j_color, j_val)

    def calculate_score(self, group):
        return sum(t["virtual_val"] for t in group)

    # ==========================================
    # 2. ÇİFT (PAIR) BULMA
    # ==========================================
    def find_pairs(self, normal_tiles, wildcards):
        pairs = []
        used_ids = set()
        
        # 1. Doğal Çiftler
        sorted_tiles = sorted(normal_tiles, key=lambda x: (x['color'], x['val']))
        for i in range(len(sorted_tiles) - 1):
            t1 = sorted_tiles[i]
            t2 = sorted_tiles[i+1]
            
            if t1['id'] in used_ids or t2['id'] in used_ids: continue
            
            if t1['color'] == t2['color'] and t1['val'] == t2['val']:
                pairs.append([t1, t2])
                used_ids.add(t1['id'])
                used_ids.add(t2['id'])
        
        # 2. Jokerli Çiftler (Tek tek eşleştir)
        if wildcards:
            leftovers = [t for t in sorted_tiles if t['id'] not in used_ids]
            leftovers.sort(key=lambda x: x['val'], reverse=True)
            
            wc_idx = 0
            for t in leftovers:
                if wc_idx >= len(wildcards): break # Joker bittiyse dur
                
                ghost = wildcards[wc_idx].copy()
                ghost['virtual_val'] = t['val']
                ghost['virtual_str'] = "(OKEY)"
                
                pairs.append([t, ghost])
                wc_idx += 1
                
        return pairs

    # ==========================================
    # 3. TÜM OLASILIKLARI ÜRET
    # ==========================================
    def get_all_candidates(self, normal_tiles, wildcards):
        candidates = []
        
        # A) NORMAL PERLER
        runs_pure = self.find_pure_runs(normal_tiles)
        sets_pure = self.find_pure_sets(normal_tiles)
        candidates.extend(runs_pure)
        candidates.extend(sets_pure)

        # B) JOKERLİ PERLER (Sadece Joker varsa girer)
        if len(wildcards) > 0:
            runs_wild = self.find_wildcard_runs(normal_tiles, wildcards)
            sets_wild = self.find_wildcard_sets(normal_tiles, wildcards)
            candidates.extend(runs_wild)
            candidates.extend(sets_wild)

        return candidates

    def find_pure_runs(self, tiles):
        runs = []
        by_color = defaultdict(list)
        for t in tiles: by_color[t["color"]].append(t)
        
        for col, lst in by_color.items():
            lst.sort(key=lambda x: x["val"])
            ext_list = self._add_14s(lst)
            curr = [ext_list[0]]
            for i in range(1, len(ext_list)):
                if ext_list[i]["val"] == curr[-1]["val"] + 1:
                    curr.append(ext_list[i])
                elif ext_list[i]["val"] == curr[-1]["val"]:
                    continue
                else:
                    if len(curr) >= 3: runs.append(self._fix_14s(curr))
                    curr = [ext_list[i]]
            if len(curr) >= 3: runs.append(self._fix_14s(curr))
        return runs

    def find_pure_sets(self, tiles):
        sets = []
        by_val = defaultdict(list)
        for t in tiles: by_val[t["val"]].append(t)
        for val, lst in by_val.items():
            uniqs = list({t["color"]: t for t in lst}.values())
            if len(uniqs) >= 3:
                sets.append(uniqs)
                if len(uniqs) == 4:
                    for c in itertools.combinations(uniqs, 3):
                        sets.append(list(c))
        return sets

    # --- JOKERLİ SERİLER ---
    def find_wildcard_runs(self, normal_tiles, wildcards):
        generated = []
        
        by_color = defaultdict(list)
        for t in normal_tiles: by_color[t["color"]].append(t)
        
        for col, lst in by_color.items():
            lst.sort(key=lambda x: x["val"])
            ext_list = self._add_14s(lst)
            
            # --- TEK JOKER KULLANIMI ---
            for joker_idx, okey_stone in enumerate(wildcards):
                for i in range(len(ext_list) - 1):
                    t1 = ext_list[i]
                    for j in range(i+1, min(i+4, len(ext_list))): 
                        t2 = ext_list[j]
                        diff = t2["val"] - t1["val"]
                        
                        # 1. ARA BOŞLUK (3 _ 5) -> Tek Joker
                        if diff == 2:
                            ghost = okey_stone.copy()
                            ghost["virtual_val"] = t1["val"] + 1
                            ghost["virtual_str"] = f"(OKEY)"
                            generated.append(self._fix_14s([t1, ghost, t2]))
                    
                        # 2. YAN YANA (10 11) -> (10 11 OKEY)
                        if diff == 1:
                            if t2["val"] < 14:
                                g_tail = okey_stone.copy()
                                g_tail["virtual_val"] = t2["val"] + 1
                                g_tail["virtual_str"] = f"(OKEY)"
                                generated.append(self._fix_14s([t1, t2, g_tail]))
                            if t1["val"] > 1:
                                g_head = okey_stone.copy()
                                g_head["virtual_val"] = t1["val"] - 1
                                g_head["virtual_str"] = f"(OKEY)"
                                generated.append(self._fix_14s([g_head, t1, t2]))

            # --- ÇİFT JOKER KULLANIMI (SADECE 2+ JOKER VARSA) ---
            if len(wildcards) >= 2:
                j_pairs = list(itertools.combinations(wildcards, 2))
                for i in range(len(ext_list) - 1):
                    t1 = ext_list[i]
                    for j in range(i+1, min(i+5, len(ext_list))):
                        t2 = ext_list[j]
                        diff = t2["val"] - t1["val"]
                        
                        # Arada 2 boşluk (3 _ _ 6)
                        if diff == 3:
                            for (j1, j2) in j_pairs:
                                g1 = j1.copy(); g1["virtual_val"] = t1["val"] + 1; g1["virtual_str"] = "(OKEY)"
                                g2 = j2.copy(); g2["virtual_val"] = t1["val"] + 2; g2["virtual_str"] = "(OKEY)"
                                generated.append(self._fix_14s([t1, g1, g2, t2]))

        return generated

    def find_wildcard_sets(self, normal_tiles, wildcards):
        generated = []
        by_val = defaultdict(list)
        for t in normal_tiles: by_val[t["val"]].append(t)
        
        for val, lst in by_val.items():
            uniqs = list({t["color"]: t for t in lst}.values())
            
            # Tek Jokerli
            if len(uniqs) >= 2:
                for joker in wildcards:
                    ghost = joker.copy()
                    ghost["virtual_val"] = val
                    ghost["virtual_str"] = "(OKEY)"
                    generated.append(uniqs + [ghost])
                        
            # Çift Jokerli (SADECE 2+ JOKER VARSA)
            if len(uniqs) >= 1 and len(wildcards) >= 2:
                j_pairs = list(itertools.combinations(wildcards, 2))
                for (j1, j2) in j_pairs:
                    t = uniqs[0]
                    g1 = j1.copy(); g1["virtual_val"] = val; g1["virtual_str"] = "(OKEY)"
                    g2 = j2.copy(); g2["virtual_val"] = val; g2["virtual_str"] = "(OKEY)"
                    generated.append([t, g1, g2])
        return generated

    def _add_14s(self, lst):
        res = lst.copy()
        for t in lst:
            if t["val"] == 1:
                copy_t = t.copy()
                copy_t["val"] = 14
                res.append(copy_t)
        res.sort(key=lambda x: x["val"])
        return res

    def _fix_14s(self, group):
        fixed = []
        for t in group:
            if t["val"] == 14:
                t_fix = t.copy()
                t_fix["val"] = 1
                if t_fix.get("is_wildcard"): t_fix["virtual_val"] = 1
                fixed.append(t_fix)
            else:
                fixed.append(t)
        return fixed

    # ==========================================
    # 4. EN İYİ ELİ BUL
    # ==========================================
    def find_best_hand(self, detected_tiles, indicator_tile):
        normal, wildcards, (j_col, j_val) = self.parse_tiles(detected_tiles, indicator_tile)
        
        candidates = self.get_all_candidates(normal, wildcards)
        
        # Puanlama Stratejisi: Joker içeren ellere öncelik ver
        def weighted_score(group):
            raw_score = self.calculate_score(group)
            joker_count = sum(1 for t in group if t.get("is_wildcard"))
            return raw_score + (joker_count * 500) 

        candidates.sort(key=weighted_score, reverse=True)
        best_hand, total_score = self._solve(candidates, [])
        pairs = self.find_pairs(normal, wildcards)
        
        return {
            "best_hand": best_hand,
            "score": total_score,
            "joker_info": f"{j_col} {j_val}",
            "pairs": pairs,
            "pair_count": len(pairs)
        }

    def _solve(self, candidates, used_ids):
        best_score = 0
        best_groups = []
        for i, group in enumerate(candidates):
            g_ids = {t["id"] for t in group}
            if not g_ids.isdisjoint(used_ids): continue
            
            current_score = self.calculate_score(group)
            new_used = used_ids + list(g_ids)
            sub_groups, sub_score = self._solve(candidates[i+1:], new_used)
            
            total = current_score + sub_score
            if total > best_score:
                best_score = total
                best_groups = [group] + sub_groups
        return best_groups, best_score