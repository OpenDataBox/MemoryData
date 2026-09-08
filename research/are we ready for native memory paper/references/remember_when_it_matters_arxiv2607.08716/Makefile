all: paper colm

paper:
	latexmk --shell-escape --pdf paper.tex

colm:
	latexmk --shell-escape --pdf colm2026_conference.tex

clean:
	rm -f paper.pdf colm2026_conference.pdf *.aux *.log *.snm *.nav *.fls *.bbl *.blg *.fdb_latexmk *.out *.toc *.brf *.bcf *.run.xml *.idx *.ilg *.ind
