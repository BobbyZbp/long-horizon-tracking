from typing import List, Tuple

def quantile(vals: List[int], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    pos = (len(vals)-1) * q
    lo = int(pos)
    hi = min(lo+1, len(vals)-1)
    if lo == hi:
        return float(vals[lo])
    w = pos - lo
    return float(vals[lo] * (1-w) + vals[hi] * w)

def gap_level_from_quantiles(gap: int, q50: float, q75: float) -> str:
    if gap <= q50:
        return "medium"
    if gap <= q75:
        return "long"
    return "extreme"

def final_sort_key(gap: int, action: float, visual: float, state: float) -> float:
    # LLM-driven, not stopwords/feature engineering
    return 2.0 * gap + 50.0 * action + 30.0 * visual + 20.0 * state