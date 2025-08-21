from __future__ import annotations

from io import StringIO
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import re
import math
import pandas as pd
from bs4 import BeautifulSoup
import json
from typing import Tuple


SUPERSUB_RE = re.compile(r"[\u00AA\u00B2\u00B3\u00B9\u2070-\u209F]")  # ª ² ³ ¹ and superscript/subscript block


def html_to_table_blocks(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    return [str(t) for t in soup.find_all("table")]

def read_tables_from_html(html: str, keep_flat_headers: bool) -> List[pd.DataFrame]:
    # Wrap literal HTML to avoid deprecation warning
    sio = StringIO(html)
    try:
        dfs = pd.read_html(sio, flavor="lxml", header=[0, 1])
    except Exception:
        sio = StringIO(html)
        dfs = pd.read_html(sio, flavor="lxml")
    if keep_flat_headers:
        return [flatten_headers(df) for df in dfs]
    return dfs

def flatten_headers(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in df.columns.values
        ]
    df.columns = [canonical_colname(c) for c in df.columns]
    return df

def canonical_colname(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name)).strip()
    s = re.sub(r"[^\w\s\-/%\(\)\+\.]", "", s)
    s = s.replace(" ", "_").lower().strip("_")
    return s

def _is_numberish(val: Any) -> bool:
    if isinstance(val, (int, float, np.number)) and not pd.isna(val):
        return True
    if isinstance(val, str):
        v = val.replace(",", "").strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", v):
            return True
        if re.fullmatch(r"-?\d+(\.\d+)?%", v):
            return True
    return False

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric-looking strings to numbers without raising deprecation warnings.
    - Strips commas
    - Turns (1.2) into -1.2
    - Converts percentages to 0-1 floats
    Leaves non-numeric cells as original strings.
    """
    for c in df.columns:
        ser = df[c]

        # Work only on object-like columns to avoid clobbering
        if not (ser.dtype == object or pd.api.types.is_string_dtype(ser)):
            continue

        s = ser.astype(str)

        # normalize formatting
        s = s.str.replace(",", "", regex=False)
        s = s.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1.2) -> -1.2

        # handle percentages first
        pct_mask = s.str.match(r"^-?\d+(\.\d+)?%$")
        s_pct = s.where(~pct_mask, s.str.rstrip("%"))
        num_pct = pd.to_numeric(s_pct, errors="coerce")
        num_pct = num_pct.where(~pct_mask, num_pct / 100.0)

        # for non-percent values
        num_plain = pd.to_numeric(s, errors="coerce")

        # combine: prefer percent-converted for pct cells, else plain
        num = num_plain.where(~pct_mask, num_pct)

        # keep original strings where conversion failed
        df[c] = num.where(~num.isna(), ser)
    return df

def choose_orientation(df: pd.DataFrame) -> str:
    """
    Heuristic:
    - If first column is mostly text-like labels and the rest are mostly number-like, use row orientation.
    - Else column orientation.
    """
    if df.shape[1] < 2:
        return "row"

    first = df.iloc[:, 0]
    rest = df.iloc[:, 1:]

    # first column text-likeness
    text_like = first.astype(str).apply(lambda x: not _is_numberish(x))
    text_ratio = float(text_like.mean())

    # numeric-ness of the remaining cells
    if rest.size:
        num_mask = rest.map(_is_numberish)  # elementwise on DataFrame
        # convert True/False to 1/0 and take mean safely
        numeric_ratio = float(np.asarray(num_mask.values, dtype=float).mean())
    else:
        numeric_ratio = 0.0

    if text_ratio >= 0.6 and numeric_ratio >= 0.6:
        return "row"
    return "column"

def df_records_rowwise(df: pd.DataFrame, col_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Row orientation:
    - first column is the label field called 'label'
    - remaining columns become key value pairs
    """
    if df.shape[1] == 0:
        return []
    out = []
    keys = [canonical_colname(x) for x in col_labels[1:]]
    for _, row in df.iterrows():
        rec: Dict[str, Any] = {"label": str(row.iloc[0])}
        for k, v in zip(keys, row.iloc[1:]):
            rec[k] = None if (isinstance(v, float) and math.isnan(v)) else v
        out.append(rec)
    return out

def df_records_columnwise(df: pd.DataFrame, col_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Column orientation:
    - one record per column
    - 'column' is the column label
    - 'values' is list of cell values in order
    - if a text label column exists and looks like labels, include it as 'index'
    """
    records: List[Dict[str, Any]] = []
    index = df.iloc[:, 0].astype(str).tolist() if df.shape[1] >= 2 else list(range(len(df)))
    for i, col in enumerate(df.columns):
        label = col_labels[i] if i < len(col_labels) else str(col)
        values = df.iloc[:, i].tolist()
        values = [None if (isinstance(v, float) and math.isnan(v)) else v for v in values]
        rec: Dict[str, Any] = {"column": str(label), "values": values}
        if i != 0 and df.shape[1] >= 2:
            rec["index"] = index
        records.append(rec)
    return records

def infer_columns_labels(df: pd.DataFrame) -> List[str]:
    """Return flat string labels for columns, even if the df has a MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        labels = []
        for tup in df.columns:
            parts = [str(x) for x in tup if str(x) != "nan"]
            label = " ".join(parts).strip()
            labels.append(label if label else "")
        return labels
    return [str(c) for c in df.columns]

def df_to_markdown(df: pd.DataFrame) -> str:
    """Consistent markdown output for saving alongside CSV."""
    return df.to_markdown(index=False)

def make_table_preview(df: pd.DataFrame, col_labels: List[str], orientation: str, max_rows: int = 6) -> Dict[str, Any]:
    """
    Small preview to embed inside elements.json so you can see structure at a glance.
    """
    prev_df = df.copy().head(max_rows)
    if orientation == "row":
        records = df_records_rowwise(prev_df, col_labels)
    else:
        records = df_records_columnwise(prev_df, col_labels)
    return {
        "orientation": orientation,
        "columns": col_labels,
        "preview_records": records,
        "preview_rows": min(len(df), max_rows),
        "total_rows": int(df.shape[0]),
        "total_cols": int(df.shape[1]),
    }

def extract_caption_neighbors(elements, table_index: int) -> Optional[str]:
    md = elements[table_index].metadata
    page = getattr(md, "page_number", None)
    j = table_index - 1
    while j >= 0:
        e = elements[j]
        if getattr(e.metadata, "page_number", None) != page:
            break
        k = e.category.lower()
        t = (e.text or "").strip()
        if k in {"title", "narrativetext"} and re.search(r"(table\s*\d+|caption)", t, re.I):
            return t
        j -= 1
    return None

def clean_label(text: str) -> str:
    s = str(text)
    # remove obvious symbols that creep in from PDF text
    s = s.replace("\u00ae", "")  # ®
    s = s.replace("\u2122", "")  # ™
    # remove superscript/subscript codepoints only
    s = SUPERSUB_RE.sub("", s)
    # collapse whitespace, keep punctuation and normal letters intact
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_mark(val: Any) -> Any:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.fullmatch(r"[Xx][\-\s]?([ct])", s)  # X-c, X t, etc.
    if m:
        return {"value": True, "qualifier": m.group(1)}
    if re.fullmatch(r"[Xx](xX)?[’']?", s):
        return True
    return val

def split_group_and_name(col_label: str) -> Tuple[str, str]:
    s = col_label.strip()
    for g in ["Treatment", "Follow-Up", "Follow-up Phone Call"]:
        if s.startswith(g + " "):
            return g, s[len(g):].strip()
        if s == g:
            return g, g
    m = re.match(r"(.+?)\s+([Vv]\d+.*|EOS.*)", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", s

def build_by_column(df: pd.DataFrame, col_labels: List[str]) -> Dict[str, Any]:
    """
    Approach 1: group -> subvisit -> {row_label: value}
    Assumes the first column holds row labels.
    """
    out: Dict[str, Any] = {}
    if df.shape[1] == 0:
        return out

    row_labels = [clean_label(x) for x in df.iloc[:, 0].tolist()]

    for j in range(1, df.shape[1]):
        col_name = col_labels[j] if j < len(col_labels) else str(df.columns[j])
        group, sub = split_group_and_name(col_name)
        group = group or "Ungrouped"
        sub = clean_label(sub)
        bucket = out.setdefault(group, {})
        rec = bucket.setdefault(sub, {})
        col_vals = df.iloc[:, j].tolist()
        for rl, v in zip(row_labels, col_vals):
            rec[clean_label(rl)] = normalize_mark(v)
    return out

def build_by_row(df: pd.DataFrame, col_labels: List[str]) -> Dict[str, Any]:
    """
    Approach 2: row_label -> group -> subvisit -> value
    """
    out: Dict[str, Any] = {}
    if df.shape[1] == 0:
        return out

    row_labels = [clean_label(x) for x in df.iloc[:, 0].tolist()]

    for i, rl in enumerate(row_labels):
        out.setdefault(rl, {})
        for j in range(1, df.shape[1]):
            col_name = col_labels[j] if j < len(col_labels) else str(df.columns[j])
            group, sub = split_group_and_name(col_name)
            group = group or "Ungrouped"
            sub = clean_label(sub)
            out[rl].setdefault(group, {})
            out[rl][group][sub] = normalize_mark(df.iat[i, j])
    return out
