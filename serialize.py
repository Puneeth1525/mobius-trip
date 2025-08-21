# serialize.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from normalize import sort_elements, stable_element_id
from tables import (
    read_tables_from_html,
    coerce_numeric,
    df_to_markdown,
    extract_caption_neighbors,
    infer_columns_labels,
    choose_orientation,
    df_records_rowwise,
    df_records_columnwise,
    make_table_preview,
    build_by_column,
    build_by_row,
)
from narrative import group_narrative
from schema import DocOut, TableOut


def export_document(
    elements: List,
    source_path: str,
    out_dir: str,
    keep_flat_headers: bool = False,
) -> DocOut:
    elements = sort_elements(elements)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # First pass: we will build enriched raw dump
    raw: List[Dict[str, Any]] = []

    tables: List[TableOut] = []
    for idx, el in enumerate(elements):
        md = getattr(el, "metadata", None)
        cat = el.category
        page = getattr(md, "page_number", None) if md else None

        entry: Dict[str, Any] = {
            "category": cat,
            "text": el.text,
            "page_number": page,
            "text_as_html": getattr(md, "text_as_html", None) if md else None,
        }

        # If this is a Table, enrich with a compact preview for elements.json
        if cat.lower() == "table":
            html = entry["text_as_html"]
            if html:
                dfs = read_tables_from_html(html, keep_flat_headers=keep_flat_headers)
                # Some table HTML may contain more than one actual table
                previews = []
                for df in dfs:
                    df = coerce_numeric(df)
                    col_labels = infer_columns_labels(df)
                    orientation = choose_orientation(df)
                    previews.append(make_table_preview(df, col_labels, orientation, max_rows=6))
                entry["table_preview"] = previews  # compact previews for quick inspection

        raw.append(entry)

    # Write enriched elements.json
    (out_root / "elements.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Second pass: materialize table artifacts and schema-like JSON
    for idx, el in enumerate(elements):
        if el.category.lower() != "table":
            continue
        md = el.metadata
        html = getattr(md, "text_as_html", None)
        if not html:
            continue

        dfs = read_tables_from_html(html, keep_flat_headers=keep_flat_headers)
        page = getattr(md, "page_number", None)
        caption = extract_caption_neighbors(elements, idx)

        # For each concrete table found inside the HTML
        for df in dfs:
            df = coerce_numeric(df)
            col_labels = infer_columns_labels(df)
            # stable id seeded from data
            elem_id = stable_element_id("table", page, None, df.to_csv(index=False))

            # Write CSV and Markdown
            csv_path = out_root / f"{elem_id}.csv"
            md_path = out_root / f"{elem_id}.md"
            df.to_csv(csv_path, index=False)
            md_str = df_to_markdown(df)
            md_path.write_text(md_str, encoding="utf-8")

            # Full JSON records – row wise and column wise
            row_records = df_records_rowwise(df, col_labels)
            col_records = df_records_columnwise(df, col_labels)
            row_json_path = out_root / f"{elem_id}.raw_rows.json"
            col_json_path = out_root / f"{elem_id}.raw_cols.json"
            row_json_path.write_text(json.dumps(row_records, ensure_ascii=False, indent=2), encoding="utf-8")
            col_json_path.write_text(json.dumps(col_records, ensure_ascii=False, indent=2), encoding="utf-8")

            # NEW hierarchical shapes
            by_col_obj = build_by_column(df, col_labels)
            by_row_obj = build_by_row(df, col_labels)
            by_col_path = out_root / f"{elem_id}.refined_by_column.json"
            by_row_path = out_root / f"{elem_id}.refined_by_row.json"
            by_col_path.write_text(json.dumps(by_col_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            by_row_path.write_text(json.dumps(by_row_obj, ensure_ascii=False, indent=2), encoding="utf-8")

            tables.append({
                "id": elem_id,
                "page": page,
                "caption": caption,
                "html_original": html,
                "csv_path": csv_path.name,
                "md_path": md_path.name,
                "row_records_path": row_json_path.name,
                "col_records_path": col_json_path.name,
                "by_column_path": by_col_path.name,  # new
                "by_row_path": by_row_path.name,  # new
                "nrows": int(df.shape[0]),
                "ncols": int(df.shape[1]),
                "headers": [str(c) for c in col_labels],
            })

    sections = group_narrative(elements)

    doc: DocOut = {
        "source_path": source_path,
        "filetype": Path(source_path).suffix.lower().lstrip("."),
        "tables": tables,
        "sections": [
            {"heading": s["heading"], "page": s["page"], "paragraphs": s["paras"]}
            for s in sections
        ],
    }
    (out_root / "document.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc
