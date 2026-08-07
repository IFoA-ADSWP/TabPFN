#!/usr/bin/env python3
"""Clone a template notebook into a new experiment notebook, output-free.

Usage:
    python scripts/infra/make_notebook.py SRC.ipynb DST.ipynb [--title "New Title"]

Keeps structure and code (imports, data loading, output conventions) from SRC,
clears outputs/execution counts/metadata, and optionally renames the first
markdown cell. Run with no args to self-check.
"""
import argparse
import pathlib
import sys
import tempfile

import nbformat


def clone(src: str, dst: str, title: str | None = None) -> None:
    nb = nbformat.read(src, as_version=4)
    nb.metadata = {}
    for cell in nb.cells:
        cell["outputs"] = []
        cell["execution_count"] = None
        cell["metadata"] = {}
    if title:
        for cell in nb.cells:
            if cell.get("cell_type") == "markdown":
                cell["source"] = f"# {title}"
                break
        else:
            nb.cells.insert(0, nbformat.v4.new_markdown_cell(f"# {title}"))
    nbformat.write(nb, dst)


def demo() -> None:
    src = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Old Title"),
        nbformat.v4.new_code_cell("print('hi')"),
    ])
    src.cells[1]["outputs"] = [nbformat.v4.new_output("stream", name="stdout", text="hi\n")]
    src.cells[1]["execution_count"] = 3
    with tempfile.TemporaryDirectory() as d:
        t, out = pathlib.Path(d) / "t.ipynb", pathlib.Path(d) / "o.ipynb"
        nbformat.write(src, t)
        clone(str(t), str(out), title="New Title")
        nb = nbformat.read(out, as_version=4)
        assert nb.cells[0]["source"] == "# New Title"
        assert nb.cells[1]["outputs"] == [] and nb.cells[1]["execution_count"] is None
        assert nb.metadata == {}
    print("self-check ok")


def main() -> None:
    if len(sys.argv) == 1:
        demo()
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--title", help="replaces the first markdown cell")
    args = ap.parse_args()
    if pathlib.Path(args.dst).exists():
        sys.exit(f"refusing to overwrite existing {args.dst}")
    clone(args.src, args.dst, args.title)


if __name__ == "__main__":
    main()
