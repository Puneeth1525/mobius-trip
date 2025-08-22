from __future__ import annotations
from typing import TypedDict, List, Optional

class TableOut(TypedDict):
    id: str
    page: int | None
    caption: Optional[str]
    html_original: str
    csv_path: str
    md_path: str
    nrows: int
    ncols: int
    headers: List[str]

class SectionOut(TypedDict):
    heading: Optional[str]
    page: int | None
    paragraphs: List[dict]

class DocOut(TypedDict):
    source_path: str
    filetype: str
    tables: List[TableOut]
    sections: List[SectionOut]


