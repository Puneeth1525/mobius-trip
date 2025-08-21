from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx

def partition_any(path: str) -> List:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return partition_pdf(
            filename=str(p),
            strategy="hi_res",
            infer_table_structure=True,
            skip_infer_table_types=False,
            # ocr_languages="eng",  # enable for scanned docs
        )
    if ext == ".docx":
        return partition_docx(filename=str(p))
    raise ValueError("Only PDF and DOCX supported")
