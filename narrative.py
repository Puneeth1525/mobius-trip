from __future__ import annotations
from typing import List, Dict

def group_narrative(elements: List) -> List[Dict]:
    sections = []
    current = {"heading": None, "page": None, "paras": []}
    for el in elements:
        kind = el.category.lower()
        txt = (el.text or "").strip()
        page = getattr(el.metadata, "page_number", None)
        if kind == "title":
            if current["paras"]:
                sections.append(current)
            current = {"heading": txt, "page": page, "paras": []}
        elif kind in {"narrativetext", "listitem"}:
            current["paras"].append({"page": page, "text": txt})
    if current["paras"]:
        sections.append(current)
    return sections
