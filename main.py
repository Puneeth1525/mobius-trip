import argparse
from partitioners import partition_any
from serialize import export_document

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to a PDF or DOCX")
    ap.add_argument("--out", default="out", help="Output folder")
    ap.add_argument("--keep-flat-headers", action="store_true",
                    help="Do not keep multirow headers. Flatten to single row for analytics")
    args = ap.parse_args()

    elements = partition_any(args.input)
    doc = export_document(
        elements,
        source_path=args.input,
        out_dir=args.out,
        keep_flat_headers=args.keep_flat_headers,
    )
    print(f"Wrote {args.out}/document.json with {len(doc['tables'])} tables")

if __name__ == "__main__":
    main()