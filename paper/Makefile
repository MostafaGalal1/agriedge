# Paper build.
#
#   make            -> both PDFs
#   make elsevier   -> manuscript.pdf       (elsarticle, single column)
#   make ieee       -> manuscript-ieee.pdf  (IEEEtran, two column)
#   make html       -> manuscript.html      (self-contained preview page)
#   make clean      -> remove build intermediates, keep outputs
#
# Both .tex files are thin class wrappers over the same body.tex, abstract.tex
# and preamble-common.tex, so the two versions cannot drift apart. Edit the
# content in those shared files, not in the wrappers.
#
# Tectonic fetches LaTeX packages on demand and caches them, so the first build
# needs network access and later ones do not.

TEX     := tectonic -X compile
TEXOPTS := --keep-intermediates --keep-logs

# The HTML preview needs the project venv, which lives beside the dataset
# directory rather than inside the package.
PYTHON  := ../../.venv/bin/python

SHARED  := body.tex abstract.tex preamble-common.tex refs.bib

.PHONY: all elsevier ieee html arxiv clean distclean fonts

all: elsevier ieee

elsevier: manuscript.pdf
ieee: manuscript-ieee.pdf

manuscript.pdf: manuscript.tex $(SHARED)
	$(TEX) $(TEXOPTS) manuscript.tex

manuscript-ieee.pdf: manuscript-ieee.tex $(SHARED)
	$(TEX) $(TEXOPTS) manuscript-ieee.tex

html: manuscript.html

manuscript.html: manuscript.md build_html.py
	$(PYTHON) build_html.py

# arXiv submission tarball, built from the Elsevier (single-column) version.
#
# That version is the right one to post: it is the only one of the two that
# needs no XeLaTeX, since the IEEE wrapper loads fontspec/unicode-math to force
# Times and arXiv defaults to pdfLaTeX. It also does not resemble a published
# IEEE article, and does not tie the preprint to a venue.
#
# refs.bib ships alongside manuscript.bbl even though arXiv never runs BibTeX:
# the .bbl is what arXiv typesets, and the .bib costs 5 KB and makes the
# tarball build correctly under any processor that does run it.
arxiv: manuscript.pdf
	@rm -rf arxiv-build arxiv.tar.gz && mkdir -p arxiv-build
	@cp manuscript.tex body.tex abstract.tex preamble-common.tex \
	    manuscript.bbl refs.bib arxiv-build/
	@tar -czf arxiv.tar.gz -C arxiv-build .
	@rm -rf arxiv-build
	@echo "wrote arxiv.tar.gz:"
	@tar -tzf arxiv.tar.gz | sed 's|^\./||' | grep . | sed 's/^/  /'

# IEEE requires Times. Confirm the embedded faces really are TeXGyreTermes and
# not a silent Latin Modern fallback -- the build emits harmless TU/ptm font
# warnings that look alarming but are not the thing to check.
fonts: manuscript-ieee.pdf
	@pdffonts manuscript-ieee.pdf | awk 'NR>2 {print $$1}' | sed 's/^[A-Z]*+//' | sort -u

clean:
	rm -f *.aux *.blg *.log *.out *.spl *.toc

distclean: clean
	rm -f manuscript.pdf manuscript-ieee.pdf manuscript.html *.bbl
