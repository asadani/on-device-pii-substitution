#!/bin/sh
# Compile the paper end-to-end with bibliography resolution.
# Run from the `paper/` directory.
#
# Build matrix: 1 xelatex -> 1 bibtex -> 2 xelatex (Lamport's recipe).
set -e

# Pass the source name as $1 (default: main).
# Use main_local for compiling on a minimal texlive-latex-base install;
# use main for the polished version that arXiv ships with full TeXLive.
SRC="${1:-main}"
pdflatex -interaction=nonstopmode $SRC.tex
bibtex $SRC
pdflatex -interaction=nonstopmode $SRC.tex
pdflatex -interaction=nonstopmode $SRC.tex

echo
echo "Built $SRC.pdf successfully."
