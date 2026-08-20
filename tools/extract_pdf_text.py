"""Extract plain text from a PDF using PyMuPDF (fitz).

Reproducible, low-overhead text extraction of the provided papers for citation and
algorithm transcription. Output is treated as a regenerable artifact (gitignored).

Usage:
    uv run python tools/extract_pdf_text.py <input.pdf> <output.txt>
"""
from __future__ import annotations

import sys
from pathlib import Path


def extract(pdf_path: Path, out_path: Path) -> int:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc, start=1):
        parts.append(f"\n\n===== PAGE {i} / {doc.page_count} =====\n")
        parts.append(page.get_text("text"))
    out_path.write_text("".join(parts), encoding="utf-8")
    return doc.page_count


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    pdf_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    n = extract(pdf_path, out_path)
    print(f"extracted {n} pages -> {out_path}")


if __name__ == "__main__":
    main()
