from __future__ import annotations
import hashlib
from typing import Any, List, Tuple, Optional

def norm_text(s: str | None) -> str:
    if not s:
        return ""
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")  # ligatures
    return s.strip()

def stable_element_id(kind: str, page: int | None, coords: Tuple[float, float, float, float] | None, text: str) -> str:
    h = hashlib.sha1()
    h.update((kind or "").encode())
    h.update(str(page or 0).encode())
    if coords:
        h.update(",".join(f"{x:.3f}" for x in coords).encode())
    h.update(norm_text(text).encode())
    return h.hexdigest()[:16]

def _bbox_from_coords(coords: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    Normalize various coordinate shapes to a bbox = (minx, miny, maxx, maxy).
    Supports:
      - object with .points where each point has .x/.y
      - dict with 'points' = [{x:..., y:...}, ...]
      - list/tuple of (x, y) pairs
    """
    pts = None
    if hasattr(coords, "points"):
        pts = coords.points
    elif isinstance(coords, dict) and "points" in coords:
        pts = coords["points"]
    elif isinstance(coords, (list, tuple)) and len(coords) and isinstance(coords[0], (list, tuple, dict)):
        pts = coords

    if not pts:
        return None

    xs, ys = [], []
    for p in pts:
        if hasattr(p, "x") and hasattr(p, "y"):
            xs.append(float(p.x)); ys.append(float(p.y))
        elif isinstance(p, dict) and "x" in p and "y" in p:
            xs.append(float(p["x"])); ys.append(float(p["y"]))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            xs.append(float(p[0])); ys.append(float(p[1]))
        # else: ignore weird point

    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))

def sort_elements(elements: List) -> List:
    def key(el):
        md = getattr(el, "metadata", None)
        page = getattr(md, "page_number", None) or 0
        coords = getattr(md, "coordinates", None)
        bbox = _bbox_from_coords(coords) if coords is not None else None
        if bbox:
            minx, miny, _, _ = bbox
            return (page, miny, minx)
        # No coords → keep relative order but still group by page
        return (page, 1e9, 1e9)
    return sorted(elements, key=key)
