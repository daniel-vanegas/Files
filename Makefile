# ==========================================
# Makefile for LaTeX article compilation
# (ROS2 code is independent and located in src/)
# ==========================================

# LaTeX environment variables (points to Definitions/ and images/)
TEXINPUTS := ./Definitions//:./images//:$(TEXINPUTS)
export TEXINPUTS

LATEX = pdflatex -synctex=1 -interaction=nonstopmode -file-line-error
BIBTEX = bibtex

# ==========================================
# Targets
# ==========================================

all: latex

latex:
	@echo "Compiling main.tex ..."
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex
	@echo "Article compiled successfully. Check main.pdf"

clean:
	@echo "Cleaning LaTeX temporary files..."
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.lof *.lot *.nav *.snm *.vrb *.synctex.gz *.fls *.fdb_latexmk
	@echo "Cleanup complete."

install:
	bash install.sh

# Optional: build the ROS2 workspace (assumes standard colcon layout inside src/)
ros:
	@echo "Building ROS2 workspace..."
	cd src && colcon build